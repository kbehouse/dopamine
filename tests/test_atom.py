import numpy as np

vmax = 10
num_atoms = 51
_support = np.linspace(-vmax, vmax, num_atoms)
probability_0 = [4.1333783e-07,6.5457175e-07,1.6567686e-06,2.2918243e-06,3.377243e-06,4.0696555e-06,2.7263784e-06,1.3954104e-06,3.145152e-06,3.3496385e-06,4.40481e-06,2.8586376e-06,4.133499e-06,6.468356e-06,3.5963114e-06,3.145734e-06,4.833817e-06,2.7578612e-06,3.9319875e-06,2.9770688e-06,3.2157723e-06,4.775921e-06,0.0010877025,0.0018106862,0.0012290915,0.023066701,0.20312686,0.5482966,0.22124515,4.360614e-06,4.4039743e-06,1.9864867e-06,5.646274e-06,2.8038917e-06,2.493179e-06,1.5277153e-06,4.5077018e-06,3.3639062e-06,3.359947e-06,2.4176475e-06,4.585951e-06,4.156065e-06,4.5249044e-06,2.8879515e-06,4.339051e-06,1.9217955e-06,3.4380253e-06,1.4442782e-06,1.4373419e-06,9.577436e-07,5.014087e-07]
probability_1 = [1.3481435e-06,3.0227682e-06,5.3148365e-06,5.0979766e-06,4.6055943e-06,6.9494863e-06,7.990088e-06,9.015933e-06,9.074443e-06,4.3097334e-06,9.997235e-06,1.1878216e-05,1.0291441e-05,8.324572e-06,6.7202295e-06,5.661624e-06,1.0074529e-05,1.21459e-05,7.311051e-06,1.217862e-05,1.1562154e-05,1.235152e-05,0.009659233,0.015129147,0.0023071256,0.03511929,0.22837247,0.51911026,0.1899506,1.2682531e-05,7.2346456e-06,1.3053184e-05,1.0949208e-05,1.00204115e-05,8.972067e-06,8.4884205e-06,8.718446e-06,9.958497e-06,8.479851e-06,1.4412476e-05,1.2609085e-05,8.827124e-06,6.5205422e-06,6.7642036e-06,7.0819588e-06,6.8651434e-06,5.271736e-06,2.1587775e-06,3.8374574e-06,2.7059818e-06,1.0853132e-06]
probability_2 = [7.351686e-07,4.4162775e-06,6.290297e-06,3.3592173e-06,4.416269e-06,1.1199697e-05,7.175254e-06,1.4659256e-05,1.1102181e-05,9.233554e-06,1.6017939e-05,8.867772e-06,6.182884e-06,3.4052275e-05,2.2711238e-05,1.1448203e-05,1.562103e-05,2.0352787e-05,1.31041415e-05,1.1559447e-05,2.206964e-05,1.8145716e-05,0.009452303,0.01782535,0.004044971,0.054764297,0.26924503,0.4855472,0.15857127,2.3495562e-05,1.3267868e-05,2.005451e-05,2.8686649e-05,1.2752553e-05,1.0958625e-05,2.4848729e-05,1.0376397e-05,1.2128878e-05,1.9223578e-05,1.2187049e-05,9.057898e-06,1.6884223e-05,1.2565048e-05,1.0541088e-05,7.347526e-06,9.1505035e-06,8.765052e-06,2.6946057e-06,8.434866e-06,2.4511698e-06,8.912643e-07]
probability_3 = [2.0077725e-06,2.3929972e-06,7.1032528e-06,7.5751364e-06,1.4608827e-05,8.447027e-06,8.284009e-06,1.1405698e-05,1.4593314e-05,1.1709787e-05,1.771602e-05,1.0680313e-05,2.0803584e-05,1.4209099e-05,1.12082635e-05,1.4195175e-05,1.0484689e-05,2.9592682e-05,9.2530945e-06,1.4248696e-05,1.0609068e-05,1.2296926e-05,0.0039195837,0.0073203836,0.0026398206,0.036974728,0.2368573,0.5226272,0.18911524,2.0576439e-05,2.3225124e-05,1.4897442e-05,2.7872275e-05,1.3467821e-05,1.0959562e-05,1.8423329e-05,1.5668455e-05,1.2176564e-05,1.3813313e-05,6.338263e-06,1.3051194e-05,1.7298733e-05,1.7416774e-05,7.754421e-06,1.1510205e-05,1.3019705e-05,6.709932e-06,7.1654117e-06,5.582142e-06,2.7460167e-06,2.6354799e-06]
probability_4 = [8.624141e-06,3.3425856e-06,3.5190653e-06,6.045648e-06,5.234189e-06,4.770694e-06,8.243626e-06,6.9353114e-06,5.9438535e-06,6.1189903e-06,1.1593944e-05,1.4341206e-05,7.2926396e-06,4.3580153e-06,5.609288e-06,3.5048308e-06,9.003593e-06,5.2620367e-06,5.1485017e-06,1.5879054e-05,5.3076888e-06,4.6812193e-06,0.5219975,0.4752674,9.6370495e-06,3.79131e-06,6.3450975e-06,0.0014520113,0.000949314,1.3315993e-05,8.724036e-06,1.0707347e-05,9.606182e-06,8.009046e-06,6.0676675e-06,3.7519105e-06,3.7330299e-06,9.425432e-06,5.941904e-06,6.3866837e-06,2.6580835e-06,6.597102e-06,1.2109583e-05,4.799512e-06,7.249709e-06,1.4531278e-05,8.1458065e-06,6.14069e-06,7.487669e-06,4.5838865e-06,3.1357713e-06]

s_x_p = _support * probability_0


print('_support = ', _support)

print('probability_0')
print('\tsum = ', np.sum(probability_0))
print('\ts_x_p_sum =' , np.sum(s_x_p))


