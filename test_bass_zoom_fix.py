#!/usr/bin/env python3
"""Test script to verify bass zoom panel display fix"""

import pygame
import numpy as np
import time

# Test window sizing calculations
def test_height_calculation():
    """Test the height calculation logic"""
    print("Testing window height calculations...")
    print("=" * 50)
    
    # Base components
    header_height = 120
    spectrum_height = 400
    spectrum_top_margin = 10
    spectrum_bottom_margin = 100
    panel_start_offset = 40
    footer_height = 80
    
    base_height = (header_height + spectrum_top_margin + spectrum_height + 
                  spectrum_bottom_margin + panel_start_offset + footer_height)
    
    print(f"Base height components:")
    print(f"  Header: {header_height}px")
    print(f"  Spectrum area: {spectrum_height}px")
    print(f"  Margins & scale: {spectrum_top_margin + spectrum_bottom_margin}px")
    print(f"  Panel offset: {panel_start_offset}px")
    print(f"  Footer: {footer_height}px")
    print(f"  Total base: {base_height}px")
    print()
    
    # Panel calculations
    panel_height = 200
    panel_spacing = 20
    
    # Test different panel combinations
    test_cases = [
        ("No panels", 0),
        ("Bass zoom only", 1),
        ("Bass zoom + Meters", 2),
        ("All 7 panels", 7)
    ]
    
    for name, panel_count in test_cases:
        if panel_count > 0:
            # Extra height for meters
            extra_height = 30 if panel_count >= 2 else 0
            total_panel_height = (panel_count * (panel_height + panel_spacing)) + extra_height
        else:
            total_panel_height = 0
            
        total_height = base_height + total_panel_height
        print(f"{name}: {total_height}px (base: {base_height}, panels: {total_panel_height})")
    
    print()
    print("Panel positioning:")
    # Calculate where panels would be drawn
    vis_start_y = header_height + spectrum_top_margin
    vis_height = spectrum_height
    first_panel_y = vis_start_y + vis_height + panel_start_offset
    
    print(f"  Spectrum starts at: y={vis_start_y}")
    print(f"  Spectrum ends at: y={vis_start_y + vis_height}")
    print(f"  First panel starts at: y={first_panel_y}")
    print(f"  Bass zoom panel height: {panel_height}px")
    print(f"  Bass zoom ends at: y={first_panel_y + panel_height}")
    print(f"  Footer starts at: y={total_height - footer_height} (for 1 panel)")

def test_visual_layout():
    """Create a visual test of the layout"""
    pygame.init()
    
    # Test window
    width = 1920
    height = 1080
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Bass Zoom Panel Layout Test")
    font = pygame.font.Font(None, 24)
    
    # Colors
    HEADER_COLOR = (40, 45, 60)
    SPECTRUM_COLOR = (20, 25, 35)
    PANEL_COLOR = (30, 35, 45)
    FOOTER_COLOR = (20, 25, 35)
    BORDER_COLOR = (100, 110, 130)
    TEXT_COLOR = (200, 210, 230)
    
    # Layout calculations (matching main app)
    header_height = 120
    spectrum_height = 400
    spectrum_top_margin = 10
    spectrum_bottom_margin = 100
    panel_start_offset = 40
    footer_height = 80
    panel_height = 200
    
    # Positions
    header_y = 0
    spectrum_y = header_height + spectrum_top_margin
    panel_y = spectrum_y + spectrum_height + panel_start_offset
    footer_y = height - footer_height
    
    running = True
    show_bass_zoom = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    show_bass_zoom = not show_bass_zoom
                    # Calculate new height
                    if show_bass_zoom:
                        required_height = header_height + spectrum_top_margin + spectrum_height + spectrum_bottom_margin + panel_start_offset + panel_height + 20 + footer_height
                    else:
                        required_height = header_height + spectrum_top_margin + spectrum_height + spectrum_bottom_margin + panel_start_offset + footer_height
                    
                    print(f"Bass zoom {'ON' if show_bass_zoom else 'OFF'}, required height: {required_height}px")
                    
                    if required_height != height:
                        height = required_height
                        screen = pygame.display.set_mode((width, height))
                        footer_y = height - footer_height
                
        screen.fill((10, 12, 15))
        
        # Draw header
        pygame.draw.rect(screen, HEADER_COLOR, (0, header_y, width, header_height))
        pygame.draw.rect(screen, BORDER_COLOR, (0, header_y, width, header_height), 2)
        text = font.render("HEADER (120px)", True, TEXT_COLOR)
        screen.blit(text, (10, header_y + 10))
        
        # Draw spectrum area
        pygame.draw.rect(screen, SPECTRUM_COLOR, (50, spectrum_y, width - 100, spectrum_height))
        pygame.draw.rect(screen, BORDER_COLOR, (50, spectrum_y, width - 100, spectrum_height), 2)
        text = font.render("SPECTRUM AREA (400px)", True, TEXT_COLOR)
        screen.blit(text, (60, spectrum_y + 10))
        
        # Draw frequency scale area
        scale_y = spectrum_y + spectrum_height + 10
        text = font.render("Frequency Scale (100px margin)", True, (150, 160, 180))
        screen.blit(text, (60, scale_y))
        
        # Draw bass zoom panel if active
        if show_bass_zoom:
            pygame.draw.rect(screen, PANEL_COLOR, (50, panel_y, width - 100, panel_height))
            pygame.draw.rect(screen, BORDER_COLOR, (50, panel_y, width - 100, panel_height), 2)
            text = font.render("BASS ZOOM PANEL (200px) - Press Z to toggle", True, TEXT_COLOR)
            screen.blit(text, (60, panel_y + 10))
            
            # Show position info
            info_text = f"Panel Y: {panel_y}, Panel bottom: {panel_y + panel_height}, Footer Y: {footer_y}"
            text = font.render(info_text, True, (100, 255, 100))
            screen.blit(text, (60, panel_y + 40))
        
        # Draw footer
        pygame.draw.rect(screen, FOOTER_COLOR, (0, footer_y, width, footer_height))
        pygame.draw.rect(screen, BORDER_COLOR, (0, footer_y, width, footer_height), 2)
        text = font.render("FOOTER (80px)", True, TEXT_COLOR)
        screen.blit(text, (10, footer_y + 10))
        
        # Instructions
        instructions = [
            "Press Z to toggle bass zoom panel",
            f"Window height: {height}px",
            f"Bass zoom: {'ON' if show_bass_zoom else 'OFF'}"
        ]
        y_offset = 10
        for inst in instructions:
            text = font.render(inst, True, (255, 255, 100))
            screen.blit(text, (width - 400, y_offset))
            y_offset += 30
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    print("Bass Zoom Panel Display Fix Test")
    print("=" * 50)
    print()
    
    # Test calculations
    test_height_calculation()
    
    print()
    print("Starting visual test...")
    print("Press Z to toggle bass zoom panel")
    print()
    
    # Visual test
    test_visual_layout()