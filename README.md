Task 2 (MPR viewer with ai):
in this project we have implemented 2 different AI models for organ detection and orientation to classify organs and clarify views (coronal , sagittal, axial and oblique) 
and visualize surfaces of segmented organs from task 1.
 
Our Project preforms the following:
•	Visualization of segmented organs surfaces with slice scrolling control.
•	Visualization of dicom files in 4 views (coronal , axial , sagittal , double oblique) .
•	Cinematic effect with FPS control and allowing the user to scroll through the different slices and navigation.
•	Friendly ROI implementation where you can limit the range of movement and cinematic effect and export availability from all views.
•	Oblique representation in other views.
•	AI detection for file orientation and top 3 organs detected. 
•	View locations movement availability.
•	Contrast control.
•	Zoom control.
•	Reset orientation and position control.


To achieve such a task we went through some steps:
1.	Creating a high quality MPR viewer
2.	Trained AI (RESNET) with 11 different patients for orientation detection
3.	Implemented AI (Total Segmentator) for organ detection
4.	Added double oblique view
5.	Added NIFTI surface visualization switchable in 4th view
6.	Implemented ROI with different export options
7.	Connected the finalized viewer with the ai model
8.	Evaluation and testing and fine tuning.
How to use: 
1.	Run the MPR Viewer app
2.	Upload your dicom folders and NIFTI file
3.	Explore the app and its features(scroll for zoom and scrolling through slices if pressed on a view, arrow keys for oblique navigation)
4.	Define your Selected ROI and choose whether you want the ROI  to Limit navigation and cinematic effect
5.	Export your ROI and choose which orientation you would like it saved as.
6.	Reupload your ROI and explore!!
AI models used:
•	Total Segmentator
•	RESNET( personally trained)
Tools used in Development:
•	Claude AI
•	VS Code


<img width="624" height="379" alt="image" src="https://github.com/user-attachments/assets/9a38ae62-c376-4898-91d6-f1df8e87b222" />
<img width="204" height="202" alt="image" src="https://github.com/user-attachments/assets/3a24382c-e2ba-45b3-a975-7b85202cf67b" />
<img width="215" height="240" alt="image" src="https://github.com/user-attachments/assets/87a4310b-c4d8-4d91-9f02-2e3bb612b2e6" />
<img width="220" height="189" alt="image" src="https://github.com/user-attachments/assets/b4238755-dfa9-4871-9373-a167405c42ba" />
<img width="208" height="213" alt="image" src="https://github.com/user-attachments/assets/3b2d316b-dfaa-409d-a7e3-ff7b748ab3c3" />
<img width="210" height="218" alt="image" src="https://github.com/user-attachments/assets/e99caca1-6778-4e48-a752-a60863cee7e7" />
<img width="180" height="221" alt="image" src="https://github.com/user-attachments/assets/43e36c20-5521-4c50-8556-75d47a9d9e57" />
<img width="624" height="202" alt="image" src="https://github.com/user-attachments/assets/d8ff5102-b8bd-4fde-be80-05f7730b9630" />



