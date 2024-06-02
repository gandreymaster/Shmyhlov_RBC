import sc
import numpy as np
import pandas as pd


I_st = 17  # number of stages
D_st = (15, 16, 17)  # number of delivery stage

# const
z = 1.645  # z_i
h = 0.45  # holding cost

cost = np.array([136.59,205.03,156.93,200,120,300,200,240,40,120,5,5,30,30,20,30,20])  # c_i
time = np.array([0,0,0,0,20,60,30,30,60,5,40,40,1,1,1,2,1])  # t_i
cumulative_cost = np.array([136.59,205.03,156.93,200,818.55,300,200,240,40,1718.55,5,5,1753.55,1753.55,1773.55,1783.55,1773.55])  # C_i
max_replenish = np.array([10,10,10,0,30,5,30,30,20,32,15,15,33,33,34,35,34])  # M_i
std_devs = np.array([152.64, 152.64,152.64,152.64,152.64,152.64,152.64,152.64,152.64,152.64, 130,80,130,80,120,50,80])  # sigma_i

node_list = np.array(range(1, I_st + 1))
D_lookup = {idx + 1: val for idx, val in enumerate(np.isin(node_list, D_st).tolist())}

L = [(1,5),(2,5), (3,5), (4,5), (5,10), (6,10), (8,10), (9,10), (7,10), (10,13), (10,14), (11,13), (13,15),(13,16), (12,14), (14,17)]
U = sc.label_stages(I_st, L)

###### forward pass #######

L = set(L)

result_array = dict()
result_array_lamb = dict()
result_array_phi = dict()

final_lambdas = dict()
final_phis = dict()

U_labeled_first_value_array = np.array([x for x, y in U])
U_labeled_second_value_array = np.array([y for x, y in U])
L_first_value_array = np.array([x for x, y in L])
L_second_value_array = np.array([y for x, y in L])
L_array = np.array(list(L))

for li in U:

    stage = li[1]
    it = stage - 1

    stream = sc.get_adjacent_nodes(L, li, U, U_labeled_first_value_array, U_labeled_second_value_array, L_array)

    x = max_replenish[it] - time[it] + 1
    y = max_replenish[it] + 1

    # if M_i < t_i
    if x < 0:
        x = y

    if D_lookup[stage]:
        y = 1

    assert x > 0

    array = np.zeros((x, y))

    for lam in range(x):
        for phi in range(y):
            array[lam, phi] = h * cumulative_cost[it] * z * np.nan_to_num(std_devs[it]) * np.sqrt(
                lam + time[it] - phi)
            for t in stream.keys():
                if stream[t] == 'down':
                    array[lam, phi] += np.nanmin(result_array_lamb[t][phi:y]) if result_array_lamb[t][
                                                                                 phi:y].size > 0 else 0
                elif stream[t] == 'up':
                    array[lam, phi] += np.nanmin(result_array_phi[t][:lam + 1])

    if D_lookup[stage]:
        lamb_cap = array.copy().flatten()
        phi_cap = np.nanmin(array, axis=1).flatten()
    else:
        lamb_cap = np.nanmin(array, axis=1).flatten()
        phi_cap = array.copy().flatten()


    result_array[stage] = array.copy()
    result_array_lamb[stage] = lamb_cap
    result_array_phi[stage] = phi_cap

    final_phis[stage] = result_array_phi[stage].argmin()
    final_lambdas[stage] = result_array_lamb[stage].argmin()

L = list(L)

for l in range(I_st - 1, 0, -1):

    label, stage = U[l - 1]
    it = stage - 1

    indices = np.where(L_second_value_array == stage)[0].tolist()

    if len(indices) > 0:

        origins = [L[i][0] for i in indices]

        ii1 = np.where(np.isin(U_labeled_second_value_array, origins))[0]
        origin_labels = [U[i][0] for i in ii1]
        correct_origins = [origins[i] for i in np.where(np.less(label, origin_labels) == True)[0]]

        for stage_to_be_processed in correct_origins:
            final_lambdas[stage] = final_phis[stage_to_be_processed]
            lambdaa = final_lambdas[stage]

            phii = np.nanargmin(result_array[stage][lambdaa, :]) if result_array[stage].shape[1] > lambdaa else 0

            final_phis[stage] = phii
    else:

        indices = np.where(L_first_value_array == stage)[0].tolist()

        destinations = [L[i][1] for i in indices]

        ii1 = np.where(np.isin(U_labeled_second_value_array, destinations))[0]
        destination_labels = [U[i][0] for i in ii1]
        correct_destinations = [destinations[i] for i in np.where(np.less(label, destination_labels) == True)[0]]

        for stage_to_be_processed in correct_destinations:

            if max_replenish[it] > final_lambdas[stage_to_be_processed]:
                final_phis[stage] = final_lambdas[stage_to_be_processed]

            lambdaa = np.nanargmin(result_array[stage][:, final_phis[stage]])

            final_lambdas[stage] = lambdaa

idx_list, lambda_list, phi_list, days_inv_list, ss_cost_list, ss_list = [], [], [], [], [], []

for a, b in zip(final_lambdas.items(), final_phis.items()):
    idx_list.append(a[0])
    lambda_list.append(int(a[1]))
    phi_list.append(b[1])
    days_inv_list.append(max(int(a[1]) + time[a[0] - 1] - b[1], 0))
    ss_cost = h * cumulative_cost[a[0] - 1] * z * std_devs[a[0] - 1] * np.sqrt(
        max(int(a[1]) + time[a[0] - 1] - b[1], 0))
    ss_cost_list.append(ss_cost)
    ss_list.append(ss_cost / (h * cumulative_cost[a[0] - 1]))

di_results = {'Index': idx_list, 'Lambda': lambda_list, 'Phi': phi_list, 'Days_of_inventory': days_inv_list,
              'Safety_stock_cost': ss_cost_list, 'Safety_stock': ss_list}

df_results = pd.DataFrame(di_results)
df_results = df_results.sort_values('Index')
print(df_results)
print(f"Total safety stock cost = {sum(df_results['Safety_stock_cost'])}")
