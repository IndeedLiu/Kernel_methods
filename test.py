import numpy as np

# Provided data
data = np.array([
    0.0089700512, 0.0013985211, 0.0086937963, 0.0015120621, 0.0014499851,
    0.0763047582, 0.0061277645, 0.0105631708, 0.0905213782, 0.0143479689,
    0.0040656766, 0.0834984182, 0.0043322335, 0.0013145536, 0.0052714409,
    0.0042013658, 0.0900538597, 0.0114157242, 0.0582217767, 0.0012394245,
    0.0013967479, 0.0617698132, 0.0042925633, 0.0116650065, 0.0013739393,
    0.0060602909, 0.0376820161, 0.0044245558, 0.0287949716, 0.0080570096,
    0.0896383536, 0.0010737985, 0.0009248030, 0.0124657438, 0.0056417297,
    0.0328411986, 0.0018017243, 0.0905213780, 0.0020542564, 0.0024493896,
    0.0615719327, 0.0040514539, 0.0013961224, 0.0021068206, 0.0035823154,
    0.0014146742, 0.0013894572, 0.0011503941, 0.0060884042, 0.0727585617,
    0.0904208834, 0.0043537663, 0.0012892232, 0.0010775026, 0.0038491604,
    0.0043680618, 0.0041354215, 0.0896444494, 0.0012740601, 0.0114208834,
    0.0763657607, 0.0043956174, 0.0261544395, 0.0036160709, 0.0039925188,
    0.0074688490, 0.0100376077, 0.0797747879, 0.0040407521, 0.0009437906,
    0.0019467588, 0.0014115931, 0.0043527486, 0.0807141366, 0.0049652291,
    0.0011591410, 0.0040221417, 0.0739745953, 0.0034968816, 0.0014005321,
    0.0040678185, 0.0019046223, 0.0016210385, 0.0009458002, 0.0013321425,
    0.0044217408, 0.0051630791, 0.0039827771, 0.0055093328, 0.0884516746,
    0.0729751083, 0.0548176151, 0.0213832889, 0.0013717585, 0.0016129617,
    0.0009251861, 0.0009912621, 0.0055200523, 0.0439750517, 0.0658239984,
    0.0010880022, 0.0013953112, 0.0011935853, 0.0267590885, 0.0673421382
])

# Calculating the mean of the first 100 values
average = np.mean(data[:100])

# Keeping two decimal places
average_rounded = round(average, 2)
print(average_rounded)