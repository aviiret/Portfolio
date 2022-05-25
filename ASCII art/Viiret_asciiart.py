from PIL import Image

# Pildi sisse lugemine
img = Image.open('keanu.jpg')

# Pildi suuruse muutmine
width, height = img.size
aspect_ratio = height / width
new_width = 120
new_height = aspect_ratio * new_width * 0.5
img = img.resize((new_width, int(new_height)))

# Pildi hallskaalasse muutmine
img = img.convert('L')
pixels = img.getdata()

# Pikslite asendamine loendis olevate sümbolitega tumeduse järgi
chars = ["B", "S", "#", "&", "@", "$", "%", "*", "!", ":", "."]
new_pixels = [chars[pixel // 25] for pixel in pixels]
new_pixels = ''.join(new_pixels)

# Sõne jagamine pildi laiusega võrdse pikkusega sõnedeks
new_pixels_count = len(new_pixels)
ascii_img = [new_pixels[index:index + new_width]
             for index in range(0, new_pixels_count, new_width)]
ascii_img = '\n'.join(ascii_img)
print(ascii_img)

# Tekstifaili kirjutamine
file = 'ascii_image.txt'
with open(file, 'w') as f:
    f.write(ascii_img)
    print('saved image to file as', file)
