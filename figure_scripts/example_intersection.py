import numpy as np
import matplotlib.pyplot as plt

def uniformPDF(interval):
    if interval[1] == interval[0]:
        return 0
    return 1 / (interval[1] - interval[0])

# Define the support intervals for R and F (uniform distributions)
support_R = (0, 1)  # Support interval for R
support_F = (.5, 1)  # Support interval for F
pdf_r = uniformPDF(support_R)
pdf_f =  uniformPDF(support_F)
print(pdf_r, pdf_f)
# Define the intersection set S
intersection_set = np.array((max(support_R[0], support_F[0]), min(support_R[1], support_F[1])))

# Define the mixture decompositions for R and F
PS_R = intersection_set  # PS for R
PLR_R = np.array((support_R[0], intersection_set[0])) # PLR for R
PS_F = intersection_set  # PS for F
PLF_F = np.array((support_F[1], intersection_set[1])) # PLF for F

alphas = np.linspace(0.1, 1, 20)
betas = [0.5]

for alpha in alphas:
    for beta in betas:
        mixture_r = beta*uniformPDF(PS_R) + (1-beta)*uniformPDF(PLR_R)
        mixture_f = alpha*uniformPDF(PS_F) + (1-alpha)*uniformPDF(PLF_F)
        if mixture_r == uniformPDF(support_R) and mixture_f == uniformPDF(support_F):
            print("IN")
        print()
        print(alpha, beta)
        print(mixture_r, mixture_f)
        print()



