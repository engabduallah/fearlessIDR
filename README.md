# fearlessIDR
The identity recognition system is implanted on the surrounding camera to detect and monitor the surrounding area around the house, representing the first protection layer. It has four  abilities: emotion detection, face recognition, dangerous weapons recognition, and matrix  barcode detection. The system is connected to the real-time database to send and receive signals  or images from the user interface. 

The emotion detection ability is reporting when the owners’ emotion is fear or any face that has 
angry emotion detecting by that one of anomalous activities. 

The face recognition ability is used to distinguish between visitors, intruders, and owners. Also, it has some predefined datasets that recognize famous criminals.
The dangerous weapons recognition ability detects anomalous activities and reports the 
breaking attempts. 

The matrix barcode detection, also known as Quick Response (QR) code ability used to 
recognize the pets as the user interface, generates unique barcodes. The system can read any 
QR code, but it will only report the unique name of the pet. In order to have a unique QR code 
for the pets, when the owner enters the pet's name in the user interface, in the backend, the 
program appends to the name 007, which acts as a special password to distinguish it from other 
QR codes. 

For example, if the user enters the name “Maya” for the pet, the generated QR code will have 
the data of “maya007,” which become a unique QR code in this case. This is to prevent any 
other generated QR code by others from being recognized by the camera. The owner should 
attach the QR code to the pet to let the camera detect it. The team decided to take this approach 
as it is hard to detect the pet’s face from a close distance, but the QE code can be printed in 
bigger size and attached to the pet, which becomes easy to detect. 
