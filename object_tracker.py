import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self):
        self.ix, self.iy, self.k = 200, 200, -1
        self.color = (0, 255, 0)
        self.c = 0
        self.old_gray = None
        self.old_pts = None
        self.mask = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix, self.iy = x, y
            self.k = 1  # Set flag to start tracking

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("draw")
        cv2.setMouseCallback("draw", self.mouse)

        # Wait until user selects a point
        while True:
            _, frm = cap.read()
            frm = cv2.flip(frm, 1)
            cv2.imshow("draw", frm)

            if cv2.waitKey(1) == 27 or self.k == 1:  # Start tracking on mouse click or ESC
                self.old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
                self.mask = np.zeros_like(frm)
                break

        cv2.destroyAllWindows()
        self.old_pts = np.array([[self.ix, self.iy]], dtype=np.float32).reshape(-1, 1, 2)

        # Begin tracking loop
        while True:
            _, new_frm = cap.read()
            new_frm = cv2.flip(new_frm, 1)
            new_gray = cv2.cvtColor(new_frm, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, new_gray, self.old_pts, None,
                maxLevel=1,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08)
            )

            key = cv2.waitKey(1)

            # Reset mask if 'e' is pressed
            if key == ord('e'):
                self.mask = np.zeros_like(new_frm)

            # Change color if 'c' is pressed
            elif key == ord('c'):
                self.color = (0, 0, 0)
                lst = list(self.color)
                self.c += 1
                lst[self.c % 3] = 255
                self.color = tuple(lst)

            # Draw lines and circles for optical flow
            for i, j in zip(self.old_pts, new_pts):
                x, y = j.ravel()
                a, b = i.ravel()
                cv2.line(self.mask, (int(a), int(b)), (int(x), int(y)), self.color, 15)

            cv2.circle(new_frm, (int(x), int(y)), 3, (255, 255, 0), 2)
            new_frm = cv2.addWeighted(new_frm, 0.8, self.mask, 0.2, 0.1)
            cv2.imshow("Tracker", new_frm)
            cv2.imshow("Drawing", self.mask)

            # Update for next frame
            self.old_gray = new_gray.copy()
            self.old_pts = new_pts.reshape(-1, 1, 2)

            # Exit if ESC is pressed
            if key == 27:
                break

        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    tracker = OpticalFlowTracker()
    tracker.run()
