gray = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_BGR2GRAY)
data = pytesseract.image_to_data(
    gray, output_type=pytesseract.Output.DICT, config="--psm 6"
)
txt = data["text"]

#####
for i, txt in enumerate(data["text"]):
    if txt.strip().upper() == "NAME":
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

plt.imshow(gray[y : y + h, x : x + w])
plt.axis("off")
plt.show()

#######

plt.imshow(gray[y1:y2, x1:x2])
plt.axis("off")
plt.show()

######

name_region = mg.extract_name_region(np.array(pages[1]), y1, y2, x1, x2)
_, name_region = cv2.threshold(name_region, 150, 255, cv2.THRESH_BINARY)

plt.imshow(name_region)
plt.axis("off")
plt.show()

######

pytesseract.image_to_string(name_region, config="--psm 7").strip()


name_region_noGrid = mg.remove_grid(name_region)

print(pytesseract.image_to_string(name_region_noGrid, config="--psm 7").strip())

plt.imshow(name_region_noGrid)
plt.axis("off")
plt.show()
