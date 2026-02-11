import pygame
import requests

pygame.init()
screen = pygame.display.set_mode([1904, 1190])
clock = pygame.time.Clock()
run = True
x, y = 0, 0
vel = 5

server_ip = "http://100.68.226.61:8000"

req = requests.post(server_ip, json={"message": "Program started"})

print(req.text)

def turn_left():
    requests.post(server_ip, json={"message": "a"})
    
def turn_right():
    requests.post(server_ip, json={"message": "d"})

def move_forward():
    requests.post(server_ip, json={"message": "w"})

def move_backward():
    requests.post(server_ip, json={"message": "s"})


while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill("green")
        pygame.draw.circle(screen, "red", (x, y), 20)
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            turn_left()
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            turn_right()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
             move_forward()
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            move_backward()
        clock.tick(60)

pygame.quit()