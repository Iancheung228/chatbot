import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
# from google.colab.patches import cv2_imshow

'''
takes in img path and parameters and extract the text from the image.
returns message which is a list of tuples, the tuple contains (a,b,c)
b is sender, c is trascribed text
'''

# for paddle
ocr = PaddleOCR(lang='ch', use_angle_cls=True)  # 'ch' = Chinese; supports English+Chinese mix

def screenshotsToText(img_path, scale=1.5):
  # === 1. Load image ===
  # img = cv2.imread("/content/Screenshot 2025-10-04 at 9.28.47 PM.png")
  img = cv2.imread(img_path)
  orig = img.copy()

  img = cv2.resize(img, None, fx=scale, fy=scale)

  # === 2. Preprocess and create masks ===
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  green_lower = np.array([35, 30, 60])
  green_upper = np.array([90, 255, 255])

  white_lower = np.array([0, 0, 240])
  white_upper = np.array([180, 50, 255])

  mask_green = cv2.inRange(hsv, green_lower, green_upper)
  mask_white = cv2.inRange(hsv, white_lower, white_upper)
  mask_all = cv2.bitwise_or(mask_green, mask_white)

  ## 🔎 Debug 1: Show color masks
  # print("green mask")
  # cv2_imshow(mask_green)
  # print("white mask")
  # cv2_imshow(mask_white)
  #cv2_imshow(mask_all)
  #cv2.waitKey(0)

  # === 3. Find contours (bubbles) ===
  contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #print("Total contours found:", len(contours))

  # debug_contours = orig.copy()
  # cv2.drawContours(debug_contours, contours, -1, (0, 0, 255), 2)
  # cv2_imshow(debug_contours)

  messages = []
  debug_img = img.copy()

  for i, c in enumerate(contours):
      x, y, w, h = cv2.boundingRect(c)

      # Filter out noise
      if w < 60 or h < 60:
          continue

      # Draw bounding boxes for debugging
      cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
      cv2.putText(debug_img, f"#{i}", (x, y - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

      bubble = img[y:y+h, x:x+w]

      # Sample mean color to classify sender
      mean_color = cv2.mean(bubble)[:3]
      hsv_mean = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
      sender = "You" if (green_lower[0] <= hsv_mean[0] <= green_upper[0]) else "Friend"

      # OCR
      # Convert to grayscale
      gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)

      # 1️⃣ Increase contrast (optional but helps a lot)
      gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)  # alpha=contrast, beta=brightness

      # 2️⃣ Apply adaptive threshold (better for variable lighting)
      thresh = cv2.adaptiveThreshold(
          gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
      )

      # 3️⃣ (Optional) Invert if text is dark on light background
      thresh = cv2.bitwise_not(thresh)

      # 4️⃣ Slight dilation to connect broken strokes (good for Chinese)
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)) #(3,3) kernel → thicker by ~1 pixel in every direction
      thresh = cv2.dilate(thresh, kernel, iterations=2)

      #print(i)
      #cv2_imshow(thresh)  # Debug: check what Tesseract will see

      # OCR 
      # 1) pytesseract
      #text = pytesseract.image_to_string(thresh, lang="chi_sim", config="--psm 6").strip()

###############################
        #2) PaddleOCR
        #PaddleOCR needs a 3-channel image.
      if len(thresh.shape) == 2:  # If grayscale, convert to BGR
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
      result = ocr.ocr(thresh)
      text = ""
      if result and len(result[0]) > 0:
          text = " ".join([line[1][0] for line in result[0]]).strip()
      #print(text)
###############################



      if text:
          messages.append((y, sender, text))

      # 🔎 Debug 2: View each cropped bubble individually (optional)
      #cv2_imshow(bubble)
      # cv2.waitKey(0)

  # === 4. Show all detected bubbles ===
  # cv2_imshow( debug_img)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  # === 5. Sort and print transcript ===
  messages.sort(key=lambda m: m[0])

  # print("\n🧠 Transcribed Conversation:\n")
  # for _, sender, text in messages:
  #     print(f"{sender}: {text}")
  return messages