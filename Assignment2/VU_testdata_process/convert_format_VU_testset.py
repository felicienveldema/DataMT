
# Line input indices
S_ID = 0
H_ID = 1
#CLICKING = 13
#BOOKING = 14
FEATURE_BEG = 2
FEATURE_END = 12
FEATURES = list(range(FEATURE_BEG, FEATURE_END+1))
#FEATURES.remove(H_ID)
num_features = len(FEATURES)


# Filenames
f_in_filename = "VU_test_data.csv"
f_out_filename = "out_VU_test_set.txt"

# Create output file
f_out = open(f_out_filename, 'w')

q=0
# Read input file line by line
with open(f_in_filename, 'r') as f:
    for line in f:
        q=q+1
        if q % 10000 == 0:
            print(q)

        x = line.strip('\n').split(',')

        rel_score = 0

        y = str(rel_score)
        y = y + " qid:" + str(x[S_ID])
        for k in range(0, num_features):
            y = y + " " + str(k+1) + ":" + str(x[FEATURES[k]])
        y = y + " #" + str(x[H_ID])

        f_out.write(y + "\n")

f_out.close()
