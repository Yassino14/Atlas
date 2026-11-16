"""
Atlas - Custom Icon & Taskbar-Pinnable Desktop Shortcut Generator (create_shortcut.py)
Generates a custom `atlas.ico` icon and creates `Atlas.lnk` on the Windows Desktop
(and in d:\\Atlas) configured so Windows 10/11 allows "Pin to taskbar".
"""

from pathlib import Path
import subprocess
import sys


def generate_atlas_icon(ico_path: Path) -> None:
    """Generate a sleek dark-theme glowing acoustic wave icon (atlas.ico) using Pillow."""
    try:
        from PIL import Image, ImageDraw

        size = 256
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Rounded dark navy background badge
        draw.rounded_rectangle(
            [(12, 12), (size - 12, size - 12)],
            radius=56,
            fill=(15, 23, 42, 255),
            outline=(56, 189, 248, 255),
            width=8,
        )

        # Inner glowing ring
        draw.ellipse(
            [(44, 44), (size - 44, size - 44)],
            outline=(14, 165, 233, 180),
            width=5,
        )

        # Acoustic equalizer bars in the center
        bars = [
            (76, 108, 148),
            (102, 80, 176),
            (128, 60, 196),
            (154, 80, 176),
            (180, 108, 148),
        ]
        for x, y_top, y_bot in bars:
            draw.rounded_rectangle(
                [(x - 7, y_top), (x + 7, y_bot)],
                radius=7,
                fill=(56, 189, 248, 255),
            )

        img.save(
            ico_path,
            format="ICO",
            sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
        )
        print(f"[Icon] Generated custom icon: {ico_path}")
    except Exception as exc:
        print(f"[Icon] Could not generate custom .ico ({exc}); using system icon fallback.")


def create_windows_shortcut() -> None:
    base_dir = Path(__file__).resolve().parent
    bat_path = base_dir / "launch_atlas.bat"
    ico_path = base_dir / "atlas.ico"

    generate_atlas_icon(ico_path)
    icon_location = str(ico_path) if ico_path.exists() else "$env:SystemRoot\\System32\\shell32.dll,220"

    # Query Windows for the exact active Desktop folder path
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True,
        text=True,
        check=False,
    )
    desktop_str = res.stdout.strip()
    desktop_dir = Path(desktop_str) if desktop_str else (Path.home() / "Desktop")

    targets = [base_dir / "Atlas.lnk"]
    if desktop_dir.exists():
        targets.append(desktop_dir / "Atlas.lnk")

    for lnk_path in targets:
        ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{str(lnk_path)}')
$Shortcut.TargetPath = "$env:SystemRoot\\System32\\cmd.exe"
$Shortcut.Arguments = '/c "{str(bat_path)}"'
$Shortcut.WorkingDirectory = '{str(base_dir)}'
$Shortcut.Description = 'Atlas - 100% Offline Voice & Acoustic Desktop Agent'
$Shortcut.IconLocation = '{icon_location}'
$Shortcut.Save()
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            check=True,
        )
        print(f"[Shortcut] Created desktop/taskbar shortcut: {lnk_path}")


if __name__ == "__main__":
    if sys.platform == "win32":
        create_windows_shortcut()
    else:
        print("Shortcut creation via WScript.Shell is only needed on Windows.")
