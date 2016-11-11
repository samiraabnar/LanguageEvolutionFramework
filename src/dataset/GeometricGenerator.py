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

            # Overpaint a smaller rectangle on the main surface
            #main_surface.fill(some_color, small_rect)
            k = 0;

            for j in np.arange(np.random.randint(2)):
                some_color = (1*np.random.random_integers(100,255), 0*np.random.random_integers(255), 0*np.random.random_integers(255))
                some_shape = [np.random.random_integers(50,255), np.random.random_integers(50,255), np.random.random_integers(50,255), np.random.random_integers(50,255)]
                pygame.draw.rect(main_surface, some_color, some_shape)
                k = 1

            if k == 0:
                # Draw an ellipse outline, using a rectangle as the outside boundaries
                some_color = (0*np.random.random_integers(255), 0*np.random.random_integers(255),1*np.random.random_integers(100,255))
                some_shape = [np.random.random_integers(50,255), np.random.random_integers(50,255), np.random.random_integers(50,55), np.random.random_integers(50,255)]
                pygame.draw.ellipse(main_surface, some_color, some_shape, 0)

            pygame.image.save(main_surface,"../single_shapes/test/image"+str(i)+".jpg")

        # Now the surface is ready, tell pygame to display it!
        pygame.display.flip()

    pygame.quit()     # Once we leave the loop, close the window.

main()