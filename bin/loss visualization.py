import matplotlib.pyplot as plt
# New data for plotting
epochs = list(range(1, 16))

# dynamic
# latest_training_loss = [
#     0.43114790320396423, 0.41376691684126854, 0.4023234285414219, 0.3937288448214531, 0.37581341341137886,
#     0.3688924126327038, 0.3688400313258171, 0.3777897357940674, 0.3843649700284004, 0.38202453777194023,
#     0.37626374140381813, 0.3737984821200371, 0.3703136555850506, 0.36622827872633934, 0.3640168197453022
# ]
# latest_validation_loss = [
#     0.4263042137026787, 0.41080959886312485, 0.4034630358219147, 0.38537802547216415, 0.37246040999889374,
#     0.37189430743455887, 0.376174695789814, 0.387962706387043, 0.3867448568344116, 0.38455383479595184,
#     0.37784378975629807, 0.3766757398843765, 0.37023698538541794, 0.3680155575275421, 0.36654817312955856
# ]

# kinematic
# Extracted training and validation losses, with validation loss divided by 2
epochs = list(range(1, 17))
latest_training_loss = [
    0.14956900477409363, 0.13526688635349274, 0.1297392100095749, 0.1450241357088089, 0.1351953148841858,
    0.13344530761241913, 0.11205421388149261, 0.14635570347309113, 0.10911525040864944, 0.11589750647544861,
    0.14249399304389954, 0.11879070103168488, 0.14557771384716034, 0.1364753544330597, 0.1190766841173172,
    0.11458957195281982
]
latest_validation_loss = [
    0.3344, 0.3059, 0.2961, 0.2936, 0.2929, 0.2930, 0.2923, 0.2931, 0.2924, 0.2938,
    0.2931, 0.2923, 0.2930, 0.2930, 0.2925, 0.2931
]
latest_validation_loss = [loss / 2 for loss in latest_validation_loss]  # Dividing validation loss by 2 because the original does not consider testloader size

# Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, extracted_training_loss, label='Training Loss', marker='o')
# plt.plot(epochs, extracted_validation_loss, label='Validation Loss (halved)', marker='o')
# plt.title('Training and Validation Loss Per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.xticks(epochs)
# plt.legend()
# plt.grid(True)
# plt.show()


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, latest_training_loss, label='Training Loss', marker='o')
plt.plot(epochs, latest_validation_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss Per Epoch of Kinematic Single Track Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.show()