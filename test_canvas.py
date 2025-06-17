#!/usr/bin/env python3
"""Test the canvas system"""

import pygame
from omega4.ui import TechnicalPanelCanvas

# Initialize pygame
pygame.init()

# Create a test window
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Canvas Test")

# Create canvas
canvas = TechnicalPanelCanvas(50, 100, 700, padding=10)

# Create mock panels
class MockPanel:
    def __init__(self, name):
        self.name = name
        self.is_frozen = False
    
    def draw(self, screen, x, y, width, height, ui_scale):
        # Draw a simple rectangle
        pygame.draw.rect(screen, (100, 100, 200), (x, y, width, height))
        font = pygame.font.Font(None, 24)
        text = font.render(self.name, True, (255, 255, 255))
        text_rect = text.get_rect(center=(x + width//2, y + height//2))
        screen.blit(text, text_rect)

# Register mock panels
mock_panels = {
    'professional_meters': MockPanel('Meters'),
    'harmonic_analysis': MockPanel('Harmonic'),
    'pitch_detection': MockPanel('Pitch'),
}

for panel_id, panel in mock_panels.items():
    canvas.register_panel(panel_id, panel)

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_m:
                is_visible = canvas.toggle_panel('professional_meters')
                print(f"Meters: {'ON' if is_visible else 'OFF'}")
            elif event.key == pygame.K_h:
                is_visible = canvas.toggle_panel('harmonic_analysis')
                print(f"Harmonic: {'ON' if is_visible else 'OFF'}")
            elif event.key == pygame.K_p:
                is_visible = canvas.toggle_panel('pitch_detection')
                print(f"Pitch: {'ON' if is_visible else 'OFF'}")
            elif event.key == pygame.K_ESCAPE:
                running = False
    
    # Clear screen
    screen.fill((20, 20, 20))
    
    # Update and draw canvas
    canvas.update(1/60.0)
    canvas.draw(screen)
    
    # Show instructions
    font = pygame.font.Font(None, 24)
    text = font.render("Press M, H, P to toggle panels. ESC to exit.", True, (200, 200, 200))
    screen.blit(text, (50, 20))
    
    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Test completed.")