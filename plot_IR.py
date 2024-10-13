import matplotlib.pyplot as plt

# Data
n_values = [100, 500, 900, 1300, 1700, 2100, 2500]
gps_IRMSE = [0.4224226, 0.1757579, 0.2031889,
             0.1737224, 0.2163332, 0.1487691, 0.1420515]
cbps_IRMSE = [0.1794470, 0.1075498, 0.1317737,
              0.1518976, 0.1207752, 0.1206499, 0.1201645]
gbm_IRMSE = [0.2153207, 0.1161363, 0.1310158,
             0.1035240, 0.1094346, 0.1026358, 0.1039087]
ebal_IRMSE = [0.1794470, 0.1075498, float('nan'), float(
    'nan'), float('nan'), float('nan'), 0.1240816]
bart_IRMSE = [0.1794470, 0.1075498, 0.1274151,
              0.1051756, 0.1053503, 0.1026779, 0.1046527]
indep_IRMSE = [0.1879687, 0.1348013, 0.1328709,
               0.1030275, 0.1171905, 0.1062534, 0.1052570]

vcnet_IRMSE = [0.24928960600093789, 0.17502859895193817, 0.10726590103730693,
               0.27172892691280365, 0.36871787998159217, 0.11532457910633953, 0.1320511509495618]
vcnet_tr_IRMSE = [0.18789898621796755, 0.16089052422112737, 0.21158275107260666,
                  0.15221343213140778, 0.3945500586530327, 0.3030381186337162, 0.14631920085156577]
drnet_tr_IRMSE = [0.24336748155590374, 0.22834981298639462, 0.2137921432448791,
                  0.24594618142157681, 0.2321527028353899, 0.24066347087136641, 0.23071917167065986]
weightednet_IRMSE = [0.15384157220816616, 0.1541057561925036, 0.20852109319304283,
                     0.25866925327236306, 0.16004866382456143, 0.09715171658879969, 0.12999174095829239]
weightednet_tr_IRMSE = [0.2162573016665402, 0.21469468499687386, 0.17545041634767605,
                        0.15881073206788607, 0.10560878093753887, 0.1380718296574029, 0.1169302420900755]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(n_values, gps_IRMSE, marker='o', label='gps')
plt.plot(n_values, cbps_IRMSE, marker='o', label='cbps')
plt.plot(n_values, gbm_IRMSE, marker='o', label='gbm')
plt.plot(n_values, ebal_IRMSE, marker='o', label='ebal')
plt.plot(n_values, bart_IRMSE, marker='o', label='bart')
plt.plot(n_values, indep_IRMSE, marker='o', label='indep')
plt.plot(n_values, vcnet_IRMSE, marker='o', label='VCNet')
plt.plot(n_values, vcnet_tr_IRMSE, marker='o', label='VCNet_TR')
plt.plot(n_values, drnet_tr_IRMSE, marker='o', label='Drnet_tr')
plt.plot(n_values, weightednet_IRMSE, marker='o', label='WeightedNet')
plt.plot(n_values, weightednet_tr_IRMSE, marker='o', label='WeightedNet_TR')

plt.xlabel('Sample Size (n)')
plt.ylabel('IRMSE')
plt.title('IRMSE Comparison Across Different Methods')
plt.legend()
plt.grid(True)

plt.show()
