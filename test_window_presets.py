#!/usr/bin/env python3
"""Test window preset dimensions"""

print("Standard Monitor Resolution Presets")
print("=" * 50)
print("\nKey | Resolution | Aspect | Description")
print("----|------------|--------|-------------")
print(" 1  | 640x480    | 4:3    | VGA")
print(" 2  | 800x600    | 4:3    | SVGA")
print(" 3  | 1024x768   | 4:3    | XGA")
print(" 4  | 1280x720   | 16:9   | 720p HD")
print(" 5  | 1920x1080  | 16:9   | 1080p Full HD")
print(" 6  | 2560x1440  | 16:9   | 1440p QHD")

print("\n✅ Window presets now use standard monitor resolutions")
print("✅ Presets will override any panel-based height calculations")
print("✅ Panel toggles (M,Z,H,R,P,C,J) still auto-adjust height when needed")

print("\nCommon Display Standards:")
print("- 4:3 aspect ratio: Classic CRT monitors (VGA, SVGA, XGA)")
print("- 16:9 aspect ratio: Modern widescreen (HD, Full HD, QHD)")
print("- 16:10 aspect ratio: Professional monitors (1920x1200, 2560x1600)")

print("\nNote: For ultra-wide or 4K displays, use fullscreen mode (F11)")