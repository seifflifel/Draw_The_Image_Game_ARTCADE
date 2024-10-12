import cv2
import numpy as np
import time

def compare_features(image_path1, image_path2, output_file='output.txt'):
    """Compare two images using feature detection, return a percentage of matches, and save the matches to a file."""
    # Load the images
    imageA = cv2.imread(image_path1)
    imageB = cv2.imread(image_path2)

    # Check if images are loaded properly
    if imageA is None or imageB is None:
        print("Error loading images. Please check the file paths.")
        return

    # Resize images to manageable size (optional, based on your need)
    imageA = cv2.resize(imageA, (800, 600))
    imageB = cv2.resize(imageB, (800, 600))

    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    keypointsA, descriptorsA = orb.detectAndCompute(grayA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(grayB, None)

    # Check if descriptors are found in both images
    if descriptorsA is None or descriptorsB is None:
        print("No descriptors found in one or both images.")
        with open(output_file, 'w') as f:
            f.write("Feature Match Percentage: 0.00%\n")
        return

    # Create a matcher and find matches
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptorsA, descriptorsB)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate the percentage of matches
    num_good_matches = len(matches)
    total_keypoints = min(len(keypointsA), len(keypointsB))  # Use the smaller number of keypoints for comparison

    # Force 100% if images are identical
    if np.array_equal(imageA, imageB):
        match_percentage = 100.0
    elif total_keypoints > 0:
        match_percentage = (num_good_matches / total_keypoints) * 100
    else:
        match_percentage = 0.0

    # Write the match percentage to the output file
    with open(output_file, 'w') as f:
        f.write(f'{match_percentage:.2f} %\n')

    print(f'Feature Match Percentage: {match_percentage:.2f} %')

    # Draw and save the matches without displaying
    match_img = cv2.drawMatches(imageA, keypointsA, imageB, keypointsB, matches[:20], None)

    # Resize the match image to a manageable size
    match_img_resized = cv2.resize(match_img, (1920, 1080))

    # Save the image of matches
    cv2.imwrite('feature_matches_output.png', match_img_resized)


def monitor_input_file(input_file, image1, image2, output_file):
    """Monitor the input file for the 'ready' state and run the comparison."""
    while True:
        # Open and read the input file
        with open(input_file, 'r') as f:
            status = f.read().strip().lower()

        # Check if the input file says 'ready'
        if status == 'ready':
            print("Input file says 'ready'. Running the comparison...")
            compare_features(image1, image2, output_file)

            # After running, change the input file to "not ready"
            with open(input_file, 'w') as f:
                f.write("not ready")
        
        # Sleep for a short while before checking the file again
        time.sleep(2)  # Check every 2 seconds

# Example usage
input_file = 'input.txt'  # The file where 'ready' or 'not ready' is written
image1 = 'pics/og1.png'  # Path to first image
image2 = 'pics/try.png'  # Path to second image
output_file = 'output.txt'  # File to write the percentage result

monitor_input_file(input_file, image1, image2, output_file)
