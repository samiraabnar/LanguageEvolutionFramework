import pygame
import numpy as np

def main():
    """ Set up the game and run the main game loop """
    pygame.init()      # Prepare the pygame module for use
    surface_sz = 480   # Desired physical surface size, in pixels.

    # Create surface of (width, height), and its window.
    main_surface = pygame.display.set_mode((surface_sz, surface_sz))

    # Set up some data to describe a small rectangle and its color
    small_rect = (300, 200, 150, 90)
    some_color = (255, 0, 0)        # A color is a mix of (Red, Green, Blue)

    while True:
        ev = pygame.event.poll()    # Look for any event
        if ev.type == pygame.QUIT:  # Window close button clicked?
            break                   #   ... leave game loop

        # Update your game objects and data structures here...
        # We draw everything from scratch on each frame.
        # So first fill everything with the background color
        for i in np.arange(1000):
            main_surface.fill((255, 149, 0))


            blue = np.random.random_integers(100,255)
            green = np.random.random_integers(0,30)
            red = np.random.random_integers(0,30)
            some_color = ( red, green , blue )
            left = np.random.random_integers(255)
            top = np.random.random_integers(255)
            length = np.random.random_integers(50,255)
            rad = np.random.random_integers(20,255)
            left = np.random.random_integers(rad,255)
            top = np.random.random_integers(rad,255)
            some_shape = [left, top,length, length]
            #pygame.draw.rect(main_surface, some_color, some_shape)
            pygame.draw.circle(main_surface,some_color,(left,top),rad,0)


            pygame.image.save(main_surface,"../blue_circle/"+"image"+str(i)+".jpg")

        # Now the surface is ready, tell pygame to display it!
        pygame.display.flip()

    pygame.quit()     # Once we leave the loop, close the window.

main()