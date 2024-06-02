import numpy as np

def choices(a_huge_key_list):
    L = len(a_huge_key_list)
    i = np.random.randint(0, L)
    return a_huge_key_list[i]

def get_adjacent_nodes(set_links, link, U_labeled, U_labeled_first_value_array,
                       U_labeled_second_value_array, L_array):
    stage = link[1]
    label = link[0]

    k, l = np.where(L_array == stage)

    inter_result = set().union(*[set_links & {(stage, j), (j, stage)} for j in
                                 np.unique(np.array([list(set_links)[i] for i in k]).flatten()).tolist() if stage != j])

    li_collect = list(inter_result)

    di_map1 = {a[0]: 'up' for a in li_collect if a[1] == stage}
    di_map2 = {a[1]: 'down' for a in li_collect if a[0] == stage}

    di_mapping = {**di_map1, **di_map2}

    candidates = [k for k in di_mapping.keys() if k != stage]
    candidate_labels = [x for x in
                        U_labeled_first_value_array[np.isin(U_labeled_second_value_array, candidates)].tolist() if
                        x < label]

    result_stages = [U_labeled[i - 1][1] for i in candidate_labels]

    return {k: v for k, v in di_mapping.items() if k in result_stages}


def get_next_nodes(array_links, array_links_first, array_links_second, ix1, ix2, stages):
    if stages.size == 1:
        return stages[0], ix1, ix2

    for j in stages:

        x1 = np.where(np.take(array_links_first, np.where(ix1)[0]) == j)[0]

        if x1.size > 0:

            list_all_stages_first = array_links[np.take(np.where(ix1)[0], x1)]
            list_all_stages_first = list_all_stages_first.flatten()

        else:
            list_all_stages_first = list()

        x2 = np.where(np.take(array_links_second, np.where(ix2)[0]) == j)[0]

        if x2.size > 0:

            list_all_stages_second = array_links[np.take(np.where(ix2)[0], x2)]
            list_all_stages_second = list_all_stages_second.flatten()

        else:
            list_all_stages_second = list()

        list_all_stages = np.concatenate((list_all_stages_first, list_all_stages_second))

        index1 = np.where(list_all_stages == j)
        list_all_connected_stages = np.delete(list_all_stages, index1)

        list_connected_unvisited_stages = list_all_connected_stages[np.isin(list_all_connected_stages, stages)]

        if list_connected_unvisited_stages.size == 1:
            ix1 = np.isin(array_links_first, j)
            ix2 = np.isin(array_links_second, j)

            return j, ix1, ix2  # Возвращаемся за пределы цикла после завершения всех итераций
        else:
            continue  # Продолжаем цикл, если еще остались непосещенные узлы


def label_stages(I, L):
    I = I
    L = L

    L_array = np.array(list(L))
    L_first_value_array = np.array([x for x, y in L])
    L_second_value_array = np.array([y for x, y in L])

    N = list(range(1, I + 1))
    U = list()
    l = 1

    N = np.array(list(reversed(sorted(N))))

    gix1, gix2 = np.zeros(L_first_value_array.size, dtype=bool), np.zeros(L_first_value_array.size, dtype=bool)

    while N.size > 0:
        x, ix1, ix2 = get_next_nodes(L_array, L_first_value_array, L_second_value_array, ~gix1, ~gix2, N)

        gix1 = np.logical_or(gix1, ix1)
        gix2 = np.logical_or(gix2, ix2)

        if x > 0:
            U.append((l, x))
            idx = np.where(N == x)
            N = np.delete(N, idx)
            l += 1
        else:
            x *= -1
            idx = np.where(N == x)
            N = np.delete(N, idx)
            N = np.append(N, x)

    return U

