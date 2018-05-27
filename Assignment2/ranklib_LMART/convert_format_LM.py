
# Line input indices
S_ID = 0
H_ID = 1
CLICKING = 13
BOOKING = 14
FEATURE_BEG = 2
FEATURE_END = 12
FEATURES = list(range(FEATURE_BEG, FEATURE_END+1))
#FEATURES.remove(H_ID)
num_features = len(FEATURES)


# Filenames
f_in_filename = "data/Val_set_LM.csv"
f_out_filename = "out_Val_set_LM.txt"

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
        if x[BOOKING] == '1':
            rel_score = 5
        elif x[CLICKING] == '1':
            rel_score = 1

        y = str(rel_score)
        y = y + " qid:" + str(x[S_ID])
        for k in range(0, num_features):
            y = y + " " + str(k+1) + ":" + str(x[FEATURES[k]])
        y = y + " #" + str(x[H_ID])

        f_out.write(y + "\n")

f_out.close()
