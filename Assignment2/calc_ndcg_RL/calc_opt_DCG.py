import math

f_in_name = "out_Test_set_LM.txt"
f_out_name = "optimal_DCGsaa.txt"

f_out = open(f_out_name, 'w')

with open(f_in_name, 'r') as f:

    # Assumes qid of -1 doesnt exist
    cur_qid = "-1"

    finished = False
    while(finished == False):

        rel_scores = []
        while(True):
            fpos = f.tell()
            line = f.readline()

            # Check for EOF
            if line == "":
                finished = True
                qid = "-1"
            else:
                x = line.strip('\n').split(' ')
                qid = x[1][4:]

            if qid != cur_qid:
                # Found the start of a new query

                # Calculate opt_DCG of cur_qid
                if cur_qid != "-1":
                    rel_scores = [r for r in rel_scores if r != 0]
                    rel_scores.sort(reverse=True)
                    DCG = 0
                    for k in range(0, len(rel_scores)):
                        DCG = DCG + (2**rel_scores[k] - 1) / math.log2(1+(k+1))
                    f_out.write(cur_qid + " " + str(DCG) + '\n')

                # Go back to start of new query
                f.seek(fpos)
                cur_qid = qid
                break

            rel_scores.append(int(x[0]))

f_out.close()







