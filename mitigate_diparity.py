import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
from sklearn.metrics import confusion_matrix
from scipy.stats import bootstrap
from metric_learn import MMC
import cmath
import sympy
import itertools
import warnings


class MLPE:

    def __init__(self):
        self.X_train_attr = None
        self.X_train_data = None
        self.X_test_attr = None
        self.X_test_data = None

        # for calculating accuracy, precision, or recall
        self.performance_test_pred = None

        self.attribute_classes = None
        self.original_attribute_classes = None
        self.n_roots = None
        self.n_imputed = None
        self.X_unity = None
        self.reference_selection = None
        self.iters = None
        self.n_balance = None

        self.sim_pair_index = None
        self.diff_pair_index = None
        self.mmc = None

        self.tree = None
        self.lattice_points = None
        self.u = None
        self.lows = None
        self.highs = None
        lattice_confidence_scores = None

    @staticmethod
    def pcomplex(theta):
        return complex(np.cos(theta), np.sin(theta))

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def prepare_data(self, data: pd.core.frame.DataFrame, demographic_attributes=None, data_cols=None):

        data = data.copy()
        # predictions are the last column
        self.predictions = data.pop(data.columns[-1])
        # labels are the penultimate column
        self.labels = data.pop(data.columns[-1])

        if demographic_attributes is None:
            demographic_attributes = data.dtypes[data.dtypes == 'O']
            demographic_attributes = list(set(demographic_attributes.index))

        if data_cols is None:
            data_cols = list(set(data.columns).symmetric_difference(set(demographic_attributes)))

        self.X_train_attr = data.loc[self.predictions.isna().values, demographic_attributes]
        self.X_test_attr = data.loc[~self.predictions.isna().values, demographic_attributes]
        self.X_train_data = data.loc[self.predictions.isna().values, data_cols]
        self.X_test_data = data.loc[~self.predictions.isna().values, data_cols]

    def remove_outliers(self, thresh=.01):
        self.original_attribute_classes = self.identify_attribute_classes()
        n = len(self.X_train_attr) * thresh
        remove_index = []
        for col in self.X_train_attr.columns:
            class_counts = self.X_train_attr[col].value_counts()
            to_exclude = list(class_counts[class_counts < n].keys())
            remove_index += list(self.X_train_attr.query(f'{col} in {to_exclude}').index)
        self.X_train_attr = self.X_train_attr.drop(remove_index, axis=0).reset_index(drop=True)

    def identify_attribute_classes(self):
        attributes = self.X_train_attr.columns
        self.attribute_classes = {attr: sorted(list(set(self.X_train_attr.loc[:, attr]))) for attr in attributes}

    def attr_to_unity(self, col):
        """
        Map each of the n attribute classes to a root of unity

        If there are not a prime number of classes, choose the next prime p's pth roots of unity
        then impute additional attributes from n to p

        """
        unity_mapping = {}
        n = len(self.attribute_classes[col])
        p = sympy.nextprime(n - 1)

        if self.n_roots is None:
            self.n_roots = {}
            self.n_imputed = {}
        self.n_roots[col] = p
        self.n_imputed[col] = p - n

        zeta = self.pcomplex(2 * np.pi / p)
        col_unities = {y[1]: zeta ** y[0] for y in enumerate(self.attribute_classes[col])}
        imputed_unities = {f'imputed_{col}_{y + 1}': zeta ** y for y in np.arange(p) if y >= n}

        return col_unities, imputed_unities

    def apply_unity_mapping(self):
        """
        Map the root of unity to each observed class within each attribute.
        For every imputed class, add column with representative root of unity
        """
        self.X_unity = pd.DataFrame(index=self.X_train_attr.index)
        for col in self.X_train_attr.columns:
            col_unities, imputed_unities = self.attr_to_unity(col)
            self.X_unity.loc[:, col] = self.X_train_attr[col].map(col_unities)
            self.X_unity.loc[:, imputed_unities.keys()] = \
                np.ones((len(self.X_unity), 1)) @ np.array(list(imputed_unities.values())).reshape(1, -1)

    def generate_permutation_powers(self):
        permutation_powers = np.array([])
        for col in self.X_train_attr.columns:
            # chooosing a random power between 1 and p to peform permuation
            power = np.random.choice(np.arange(1, self.n_roots[col]))
            # applying the power to attribute column and all associated imputed columns
            power_array = power * np.ones(self.n_imputed[col] + 1)
            permutation_powers = np.concatenate([permutation_powers, power_array])

        return permutation_powers

    def generate_imputed_scaling(self):
        # each imputed result should only appear 1/p of the time, so muliply imputed_unity by 1/p
        imputed_scaling = np.array([])
        for col in self.X_train_attr.columns:
            # chooosing a random power between 1 and p to peform permuation
            scale = 1 / self.n_roots[col]
            # applying the power to attribute column and all associated imputed columns
            scale_array = scale * np.ones(self.n_imputed[col])
            imputed_scaling = np.concatenate([imputed_scaling, np.array([1]), scale_array])
        return imputed_scaling

    def create_X_unity_adj(self, imputed_scaling, permutation_powers=None):
        """
        Apply permutation to all roots of unity if selected,
        then add sum of imputed values to each attribute and remove imputed columns
        """

        if permutation_powers is None:
            X_unity_adj = (self.X_unity * imputed_scaling).copy()
        else:
            X_unity_adj = (np.power(self.X_unity, permutation_powers) * imputed_scaling).copy()

        for col in self.attribute_classes.keys():
            X_unity_adj.loc[:, col] += X_unity_adj.filter(regex=f'imputed_{col}_\d+').sum(axis=1)

        X_unity_adj = X_unity_adj.filter(regex='^(?!imputed).+').copy()

        return X_unity_adj

    def select_balanced(self, n, d=3, iters=1000):
        self.iters = iters
        self.n_balance = n
        self.identify_attribute_classes()
        self.apply_unity_mapping()

        # initiate selection by adding an equal number of each class (within each attribute)
        ref_n = n // d
        init_indices = set()
        for attr, classes in self.attribute_classes.items():
            n_per_class = ref_n // (len(classes))
            init_indices = init_indices.union(set(self.X_train_attr.groupby(attr).head(n_per_class).index))
        selection = np.zeros(len(self.X_train_attr))
        selection[list(init_indices)] = 1

        best_obs = None
        costs = []
        imputed_scaling = self.generate_imputed_scaling()

        for j in range(iters):

            # Every interval, permute roots of unity to ensure valid optimization
            interval = 50
            if j % interval == 0:
                permutation_powers = self.generate_permutation_powers()
                X_unity_adj = self.create_X_unity_adj(imputed_scaling, permutation_powers)
            if j % (2 * interval) == 0:
                X_unity_adj = self.create_X_unity_adj(imputed_scaling)

            all_inds = np.arange(len(X_unity_adj))
            selection_ind = np.array(all_inds * selection, dtype=int)
            selection_ind = np.array(np.arange(len(X_unity_adj)) * selection, dtype=int)
            selection_ind = selection_ind[selection_ind > 0]

            cat_stdevs = 0
            pair_selection = self.X_train_attr.loc[selection_ind]
            for col in pair_selection.columns:
                cat_stdevs += pair_selection.groupby(col).count().iloc[:, :1].std().sum() ** 2

            if best_obs is None:
                best_obs = {'stdev_cost': cat_stdevs,
                            'selection_ind': selection_ind}
            elif best_obs['stdev_cost'] > cat_stdevs:
                best_obs = {'stdev_cost': cat_stdevs,
                            'selection_ind': selection_ind}
            costs.append(cat_stdevs)

            current_state = ((X_unity_adj.transpose() @ selection))
            current_cost = np.linalg.norm(np.abs((X_unity_adj.transpose() @ selection)))

            # increasing size of selection
            if selection.sum() < n / d:
                step_size = 100
                step_ind = np.random.choice(X_unity_adj.index, (step_size))
                trial_unity = np.ones((len(step_ind), 1)) @ (current_state.values).reshape(1, -1) + X_unity_adj.loc[
                    step_ind]
                trial_unity['new_cost'] = np.linalg.norm(trial_unity, axis=1)
                trial_unity['switch'] = trial_unity['new_cost'] < current_cost
                iter_indices = trial_unity[trial_unity['switch']].sort_values(by='new_cost')[:10].index
                selection[iter_indices] = 1

            # decreasing size of selection
            else:
                all_inds = np.arange(len(X_unity_adj))
                dampen = np.random.binomial(1, .25, len(selection))
                desc_step_ind = np.array(all_inds * selection * dampen, dtype=int)
                desc_step_ind = desc_step_ind[desc_step_ind > 0]

                trial_unity = np.ones((len(desc_step_ind), 1)) @ (current_state.values).reshape(1, -1) - \
                              X_unity_adj.loc[desc_step_ind]
                trial_unity['new_cost'] = np.linalg.norm(trial_unity, axis=1)
                trial_unity['switch'] = trial_unity['new_cost'] < current_cost
                iter_indices = trial_unity[trial_unity['switch']].sort_values(by='new_cost')[:10].index
                selection[iter_indices] = 0

        self.reference_selection = best_obs.get('selection_ind')

    def find_pairs(self):

        all_inds = np.arange(len(self.X_train_attr))
        other_ind = [i for i in all_inds if i not in self.reference_selection]
        other_selection = self.X_train_attr.loc[other_ind]

        # creating tree
        S = pd.concat([pd.get_dummies(self.X_train_attr.loc[self.reference_selection][col], prefix=col) \
                       for col in self.X_train_attr.columns], axis=1)
        T = pd.concat([pd.get_dummies(self.X_train_attr.loc[other_ind][col], prefix=col) \
                       for col in self.X_train_attr.columns], axis=1)

        tree = KDTree(T, leaf_size=2, metric='manhattan')

        # finding similar points
        tree_dist, tree_ind = tree.query(S, k=10)

        similarity_choices_df = self.X_train_attr.loc[other_ind].iloc[tree_ind[:, :3].flatten()]
        similarity_choices_df['reference'] = (self.reference_selection.reshape(-1, 1) * np.ones((1, 3))).flatten()
        # print((self.reference_selection.reshape(-1,1)*np.ones((1,3))).flatten())

        similarity_choices_df.columns = [f'{col}_sim' for col in similarity_choices_df.columns]
        sim_pairing = similarity_choices_df.join(self.X_train_attr.loc[self.reference_selection],
                                                 on='reference_sim')
        sim_pair_index = np.array([sim_pairing['reference_sim'].values, sim_pairing.index.values]).transpose()

        # gathering the 10th distance away from each point in reference
        nth_distance = tree_dist[:, 9]

        # grouping reference by 10th distance
        max_distances = {d: np.where(nth_distance == d) for d in set(nth_distance)}

        # for every distance find those which are precily d + 2 (around the block) away and choose 15 of them at random
        # with replacement in case there aren't enough

        different_choices = np.array([])
        reference_index = np.array([])
        for d, ind in max_distances.items():
            ind = np.array(ind).flatten()
            r1_ind = tree.query_radius(S.iloc[ind], r=d)
            r2_ind = tree.query_radius(S.iloc[ind], r=d + 2)

            for i in np.arange(len(ind)):
                just_outside_ind = np.setdiff1d(r2_ind[i], r1_ind[i])
                different_ind = np.random.choice(just_outside_ind, 15, replace=True)

                reference_ind = int(S.iloc[ind[i]].name)
                if len(different_ind) > 0:
                    different_choices = np.concatenate([different_choices, np.array(T.iloc[different_ind].index)])
                    reference_index = np.concatenate([reference_index, np.ones(15) * reference_ind])
                else:
                    all_random = np.random.choice(self.X_train_attr.index, 15)
                    different_choices = np.concatenate([different_choices, np.array(all_random)])
                    reference_index = np.concatenate([reference_index, np.ones(15) * reference_ind])

        difference_choices_df = self.X_train_attr.loc[different_choices]
        difference_choices_df.loc[:, 'reference'] = reference_index
        difference_choices_df.columns = [f'{col}_diff' for col in difference_choices_df]
        difference_pairing = difference_choices_df.join(self.X_train_attr.loc[self.reference_selection],
                                                        on='reference_diff')

        original_reference_ind = difference_pairing['reference_diff'].values
        original_diff_ind = difference_pairing.index
        difference_pairing.reset_index(drop=True, inplace=True)

        diff_pairing_obj = MLPE()
        diff_pairing_obj.X_train_attr = difference_pairing.drop('reference_diff', axis=1)
        diff_pairing_obj.select_balanced(n=self.n_balance * 3, iters=self.iters)

        # balanced_diff_ind = original_diff_ind
        diff_ref_index = original_reference_ind[diff_pairing_obj.reference_selection]
        diff_index = original_diff_ind[diff_pairing_obj.reference_selection]

        diff_pair_index = np.array([diff_ref_index, diff_index]).transpose()

        self.sim_pair_index = sim_pair_index
        self.diff_pair_index = diff_pair_index

        return None

    def describe_pairs(self):
        ref_sims, sims = self.sim_pair_index[:, 0], self.sim_pair_index[:, 1]
        ref_diffs, diffs = self.diff_pair_index[:, 0], self.diff_pair_index[:, 1]
        cols = self.X_train_attr.columns
        for col in cols:
            print(col)
            print('ref_sim\n', self.X_train_attr.loc[ref_sims, col].value_counts())
            print('\n')
            print('sim\n', self.X_train_attr.loc[sims, col].value_counts())
            print('\n')
            print('ref_diff\n', self.X_train_attr.loc[ref_diffs, col].value_counts())
            print('\n')
            print('diff\n', self.X_train_attr.loc[diffs, col].value_counts())
            print('\n')
            print('--')

    def standardize(self):
        warnings.warn(
            'You are standardizing the traning and test data.This means the data may look different than in the model')
        scaler = StandardScaler()
        data_columns = self.X_train_data.columns
        self.X_train_data = pd.DataFrame(scaler.fit_transform(self.X_train_data), columns=data_columns)
        self.X_test_data = pd.DataFrame(scaler.fit_transform(self.X_test_data), columns=data_columns)

    def project_metric_space(self, max_iter=1000, conv_thresh=.0001, max_proj=100000, verbose=True, random_state=0):

        mmc_sim_data = np.array([np.array(self.X_train_data.loc[self.sim_pair_index[:, 0]], dtype=float),
                                 np.array(self.X_train_data.loc[self.sim_pair_index[:, 1]], dtype=float)])
        mmc_sim_data = np.reshape(mmc_sim_data, (len(self.sim_pair_index), 2, -1))

        mmc_diff_data = np.array([np.array(self.X_train_data.loc[self.diff_pair_index[:, 0]], dtype=float),
                                  np.array(self.X_train_data.loc[self.diff_pair_index[:, 1]], dtype=float)])
        mmc_diff_data = np.reshape(mmc_diff_data, (len(self.diff_pair_index), 2, -1))

        mmc_data = np.concatenate([mmc_sim_data, mmc_diff_data])

        y = np.concatenate([np.ones(len(self.sim_pair_index)),
                            -np.ones(len(self.diff_pair_index))])

        mmc = MMC(max_iter=max_iter,
                  convergence_threshold=conv_thresh,
                  max_proj=max_proj,
                  verbose=True,
                  random_state=0)

        print('fitting', len(y), 'pairs')
        mmc.fit(mmc_data, y)
        print('done!')
        self.mmc = mmc

    def identify_performance_metric_indices(self, metric='recall'):
        test_pred = pd.concat([self.predictions, self.y_test], axis=1)
        test_pred.columns = ['predictions', 'y_test']
        if metric in ('recall', 'sensitivity'):
            self.performance_test_pred = test_pred[test_pred['y_test'] == 1].index
        elif metric in ('specificity'):
            self.performance_test_pred = test_pred[test_pred['y_test'] == 0].index
        elif metric in ('precision'):
            self.performance_test_pred = test_pred[test_pred['predictions'] == 1].index
        elif metric in ('accuracy'):
            self.performance_test_pred = test_pred.index

    def create_lattice(self, desired_points=100000):
        """
        Creates a high-dimentional lattice with approximatly the desired number of points.

        The structure of the lattice is chosen so that the distance between each point within a
        given lattice dimnsion is identical (within and across all dimensions). If a dimension does
        not have a large enough 1st-99th percentile spread, only one point is chosen at the mean of
        percentiles.
        """

        # transforming test-set data points from the metric learning produced linear transformation
        X = self.mmc.transform(self.X_test_data)

        # creating a K-d Tree on the transformed test set data
        tree = KDTree(X, leaf_size=2)

        # identifying the 1st and 99th percentile points in each demention in transformed metric space
        lows, highs = np.percentile(X, q=[1, 99], axis=0)

        # enforcing non-zero distance between percentiles
        epsilon = np.float_power(10, -10)
        L = (highs - lows) + epsilon

        # starting distance between points, u, is initiated at a point which assumes equal dimension sizes
        # this is an upper bound of u
        u = L.sum() / np.power(desired_points, 1 / len(L))

        upper_u = u
        lower_u = None
        for i in range(1000):
            res = np.product(np.ceil(L / u))
            # if we have a better lower bound
            if res < desired_points:
                upper_u = u
            # if we have exceeded points for the first time
            if res > desired_points:
                lower_u = u

            # if we don't have an upper bound yet, just decrease u by 1
            # otherwise, take the average of the two u's
            if lower_u is None:
                u -= 1
            else:
                u = np.mean([lower_u, upper_u])

        basis = []

        L = (highs - lows) + epsilon
        for i, n in enumerate(np.ceil(L / u)):
            offset = (L[i] - u * (n - 1)) / 2
            basis.append(list(np.arange(lows[i] + offset, highs[i] + epsilon, u)))

        lattice_points = pd.DataFrame(list(itertools.product(*basis)))

        self.tree = tree
        self.lattice_points = lattice_points
        self.u = u
        self.lows = lows
        self.highs = highs

        return basis

    def calculate_lattice_performance(self, subset=1000, r=3, confidence_level=.95):
        lattice_ci = {}
        for chunk in self.chunks(self.lattice_points.index, subset):

            # identifying the test-set points near the lattice point from the transformed metric space
            ind = self.tree.query_radius(self.lattice_points.loc[chunk], r=r)

            for i, neighbors in enumerate(ind):
                # selecting datapoints that are relevent to model performance metric
                neighbors = [x for x in neighbors if x in self.performance_test_pred]

                # collecing performance information for the neighbors
                neighbor_performance = (self.y_test.loc[neighbors] == self.predictions.loc[neighbors]).values * 1

                # adding 0, 1 to ensure confidence intervals have heterogeneity
                ci_data = np.concatenate([neighbor_performance, np.array([0, 1])])

                # creating a confidence interval from the neighboring points on the chosen metric performance metric
                ci = bootstrap((ci_data,), np.mean, confidence_level=confidence_level).confidence_interval

                # saving confidence interval data for the specific lattice point
                lattice_ci[chunk[i]] = [ci.low, ci.high]

        self.lattice_confidence_scores = pd.DataFrame.from_dict(lattice_ci, orient='index', columns=['low_ci', 'high_ci'])

    def compare_train_test_distributions(self, bootrap_size=100, confidence_level=.95):

        distance_list = []
        while len(distance_list) < boostrap_size:

            # randomly choose a sample of test set size from the training set
            train_samp = np.random.choice(self.X_train_data.index, len(self.X_test_data))

            # calculate Wasserstein distance between distributions along each feature, add together for "Manhattan distance of Wasserstein distances"
            dist = 0
            for column in self.X_train_data.columns:
                dist += wasserstein_distance(self.X_train_data.loc[train_samp, column].values,
                                             self.X_train_data.loc[:, column].values)
            distance_list.append(dist)

        test_dist = 0
        for column in self.X_train_data.columns:
            test_dist += wasserstein_distance(self.X_test_data.loc[:, column].values,
                                              self.X_train_data.loc[:, column].values)

        ci = bootstrap((distance_list,), np.mean, confidence_level=confidence_level).confidence_interval

        if test_dist < ci.low or test_dist > ci.high:
            warnings.warn(
                'feature distributions of train and test data differ significantly: epistemic uncertainty evaluation may not be accurate')

        output = {
            'test_distance': test_dist,
            'low_percentile': ci.low,
            'high_percentile': ci.high
        }

        return output

    def output_lattice(self, target='csv', path='', suffix=''):
        """
        Outputs lightweight file/object which can be used to reconstruct the lattice
        If csv is chosen, two csv's are created:
        lattice_structure.csv : lows, highs, and unit size of each dimention of lattice
        lattice_scores.csv : confidence interval for each lattice point
        """

        if target == 'csv':
            ldf = pd.DataFrame()
            ldf['lows'] = self.lows
            ldf['highs'] = self.highs
            ldf['u'] = self.u
            ldf.to_csv(f'{path}lattice_structure{suffix}.csv')

            self.lattice_confidence_scores.to_csv(f'{path}lattice_scores{suffix}.csv')

        else:
            warnings.warn('We currently do not support any other output than csv')
            pass

    def identify_lattice_score(self, record, information_source='self', path='', suffix=''):
        """
        Identifies the closest lattice point and outputs the confidence score.
        If information_source is 'self', class attributes are used
        If information_source is 'csv', outputted csv's are loaded in. Make sure to use appropriate path and suffix.
        """
        # enforces positive lattice widths
        epsilon = np.float_power(10, -10)

        if information_source == 'self':
            ldf = pd.DataFrame()
            ldf['lows'] = self.lows
            ldf['highs'] = self.highs
            ldf['u'] = self.u
        elif information_source == 'csv':
            ldf = pd.read_csv(f'{path}lattice_structure{suffix}.csv')
        else:
            warnings.warn('We currently do not support any other information sourse')
            return None

        ldf['record'] = record
        # calculating number of points per lattice dimension
        ldf['n'] = np.ceil((ldf['highs'] - ldf['lows'] + epsilon) / ldf['u'])

        # calcuating offset of first point in lattice dimension
        ldf['offset'] = (ldf['highs'] - ldf['lows'] + epsilon - ldf['u'] * (ldf['n'] - 1)) / 2

        # identifying index of closest lattice point to record in each lattice point
        ldf['i'] = np.array(
            np.clip(np.round((ldf['record'] - ldf['lows'] - ldf['offset']) / ldf['u']), 0, ldf['n'] - 1), dtype=int)

        # generating structure to identify index of point in high dimensional lattice.
        # Follows convention of itertools.product.
        ldf['base_counts'] = np.concatenate([(ldf['n'][::-1]).cumprod()[::-1], np.array([1]), ])[1:]

        lattice_index = np.dot(ldf['base_counts'], ldf['i'])

        if information_source == 'self':
            return self.lattice_confidence_scores.iloc[lattice_index]
        elif information_source == 'csv':
            lattice_confidence_scores = pd.read_csv(f'{path}lattice_structure{suffix}.csv')
            return lattice_confidence_scores.iloc[lattice_index]


