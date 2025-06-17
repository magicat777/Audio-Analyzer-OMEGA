"""
Utility functions for panel drawing and layout
"""

import pygame
from typing import Optional, Tuple


def draw_panel_header(screen: pygame.Surface, title: str, font: pygame.font.Font, 
                     x: int, y: int, width: int, height: int = 35,
                     bg_color: Tuple[int, int, int] = (20, 25, 35),
                     border_color: Tuple[int, int, int] = (70, 80, 100),
                     text_color: Tuple[int, int, int] = (200, 220, 255),
                     frozen: bool = False) -> int:
    """
    Draw a centered panel header with background
    
    Args:
        screen: Pygame surface to draw on
        title: Panel title text
        font: Font to use for title
        x: Panel x position
        y: Panel y position
        width: Panel width
        height: Header height (default 35)
        bg_color: Header background color
        border_color: Header border color
        text_color: Title text color
        frozen: Whether panel is frozen (adds [FROZEN] suffix)
        
    Returns:
        int: Y position after header (for content placement)
    """
    # Draw header background
    header_rect = pygame.Rect(x, y, width, height)
    pygame.draw.rect(screen, bg_color, header_rect)
    pygame.draw.rect(screen, border_color, header_rect, 1)
    
    # Add frozen indicator if needed
    display_title = title
    if frozen:
        display_title += " [FROZEN]"
    
    # Render and center title
    title_surface = font.render(display_title, True, text_color)
    title_rect = title_surface.get_rect()
    title_rect.centerx = x + width // 2
    title_rect.centery = y + height // 2
    
    screen.blit(title_surface, title_rect)
    
    # Draw separator line
    pygame.draw.line(screen, border_color, 
                    (x, y + height - 1), 
                    (x + width - 1, y + height - 1), 1)
    
    return y + height


def draw_panel_background(screen: pygame.Surface, x: int, y: int, 
                         width: int, height: int,
                         bg_color: Tuple[int, int, int] = (20, 25, 35),
                         border_color: Tuple[int, int, int] = (70, 80, 100),
                         alpha: int = 240):
    """
    Draw panel background with semi-transparency
    
    Args:
        screen: Pygame surface to draw on
        x: Panel x position
        y: Panel y position
        width: Panel width
        height: Panel height
        bg_color: Background color
        border_color: Border color
        alpha: Transparency (0-255)
    """
    # Semi-transparent background
    overlay = pygame.Surface((width, height))
    overlay.set_alpha(alpha)
    overlay.fill(bg_color)
    screen.blit(overlay, (x, y))
    
    # Border
    pygame.draw.rect(screen, border_color, (x, y, width, height), 2)