
def sec_2_hms(t):
    hms = '%02dh%02dm%02ds' % (t/3600, t/60 % 60  , t%60)
    return hms



steps_per_second = 10
want_steps = 5e5 #2e6  # 2M

t_str = sec_2_hms(want_steps / steps_per_second)
print(t_str)