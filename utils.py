from math import log2


# get to input "target" by multiplying output "start_n" by output "upscale" for input "steps" times
# EQ: target = start_n * (upscale ** steps), solve for start_n and upscale as whole numbers if target is a power of 2
# find minimum start_n possible
def get_params(target, steps):
    # worst-case (largest) start_n is when target is divided by 2 for "steps" times
    start_n = target / (2 ** steps)
    upscale = 2
    # try powers-of-two starting at 4 (so start counter at 2 since 2 ** 2 = 4) since 2 was already
    # calculated above
    for i in range(2, int(log2(target))):
        n = target
        for _ in range(steps):
            n /= i ** 2
        # if n is a whole number, it must be better than the current start_n since the loop
        # uses progressively bigger divisors, creating lower n's
        if n % 1 == 0:
            start_n = n
            upscale = i ** 2
    return int(start_n), int(upscale)
