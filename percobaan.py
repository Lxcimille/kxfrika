# Import libraries
import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

from deap import base, creator, tools

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# ==============================
# Preprocessing Steps
# ==============================

# 1. Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 2. Remove low-variance features (threshold = 0, i.e., zero variance)
selector_variance = VarianceThreshold(threshold=0.0)
X = selector_variance.fit_transform(X)

# 3. Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

n_features = X.shape[1]

print(f"Dataset: Breast Cancer")
print(f"Samples: {X.shape[0]}, Features after preprocessing: {X.shape[1]}")
print("=" * 60)

# Set random seeds for reproducibility
random.seed()
np.random.seed()

# Initialize results storage
results = {}

# =================================================================
# METHOD 1: GA + SFS (Your Original Hybrid Method)
# =================================================================
print("Running METHOD 1: GA + SFS Hybrid...")

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_accuracy(ind):
    if sum(ind) != 13: return 0.0,
    idx = [i for i, bit in enumerate(ind) if bit == 1]
    clf = KNeighborsClassifier()
    score = cross_val_score(clf, X[:, idx], y, cv=5).mean()
    return score,

toolbox.register("evaluate", eval_accuracy)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA execution
population = toolbox.population(n=30)
N_GEN = 20

for gen in range(N_GEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values
    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    invalid = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    population[:] = offspring

# Best GA result
best_ind = tools.selBest(population, 1)[0]
ga_selected = [i for i, b in enumerate(best_ind) if b == 1]
ga_only_score = cross_val_score(KNeighborsClassifier(), X[:, ga_selected], y, cv=10).mean()

# SFS refinement on GA results
clf = KNeighborsClassifier()
sfs = SequentialFeatureSelector(clf, direction='forward', cv=5)
sfs.fit(X[:, ga_selected], y)
ga_sfs_final_mask = np.array(ga_selected)[sfs.get_support()]
ga_sfs_score = cross_val_score(clf, X[:, ga_sfs_final_mask], y, cv=10).mean()

# Store results
results['GA Only'] = {
    'features': ga_selected,
    'n_features': len(ga_selected),
    'accuracy': ga_only_score
}

results['GA + SFS'] = {
    'features': ga_sfs_final_mask,
    'n_features': len(ga_sfs_final_mask),
    'accuracy': ga_sfs_score
}

print(f"✓ GA Only: {len(ga_selected)} features, Accuracy: {ga_only_score:.4f}")
print(f"✓ GA + SFS: {len(ga_sfs_final_mask)} features, Accuracy: {ga_sfs_score:.4f}")

# =================================================================
# METHOD 2: Pure SFS (Forward Selection from scratch)
# =================================================================
print("\nRunning METHOD 2: Pure SFS...")

clf = KNeighborsClassifier()
target_features = 13
sfs_pure = SequentialFeatureSelector(clf, direction='forward', cv=5, n_features_to_select=target_features)
sfs_pure.fit(X, y)
sfs_features = np.where(sfs_pure.get_support())[0]
sfs_score = cross_val_score(clf, X[:, sfs_features], y, cv=10).mean()

results['Pure SFS'] = {
    'features': sfs_features,
    'n_features': len(sfs_features),
    'accuracy': sfs_score
}

print(f"✓ Pure SFS: {len(sfs_features)} features, Accuracy: {sfs_score:.4f}")

# =================================================================
# METHOD 3: Pure Backward SFS
# =================================================================
print("\nRunning METHOD 3: Pure Backward SFS...")

sfs_backward = SequentialFeatureSelector(clf, direction='backward', cv=5, n_features_to_select=target_features)
sfs_backward.fit(X, y)
sfs_back_features = np.where(sfs_backward.get_support())[0]
sfs_back_score = cross_val_score(clf, X[:, sfs_back_features], y, cv=10).mean()

results['Backward SFS'] = {
    'features': sfs_back_features,
    'n_features': len(sfs_back_features),
    'accuracy': sfs_back_score
}

print(f"✓ Backward SFS: {len(sfs_back_features)} features, Accuracy: {sfs_back_score:.4f}")

# =================================================================
# METHOD 4: SelectKBest (Univariate Selection)
# =================================================================
print("\nRunning METHOD 4: SelectKBest...")

selector = SelectKBest(f_classif, k=target_features)
X_kbest = selector.fit_transform(X, y)
kbest_features = np.where(selector.get_support())[0]
kbest_score = cross_val_score(KNeighborsClassifier(), X_kbest, y, cv=10).mean()
# =================================================================
# METHOD 5: All Features (Baseline)
# =================================================================
print("\nRunning METHOD 5: All Features (Baseline)...")

all_features_score = cross_val_score(KNeighborsClassifier(), X, y, cv=10).mean()

results['All Features'] = {
    'features': list(range(n_features)),
    'n_features': n_features,
    'accuracy': all_features_score
}

print(f"✓ All Features: {n_features} features, Accuracy: {all_features_score:.4f}")

# =================================================================
# COMPARISON RESULTS
# =================================================================
print("\n" + "="*70)
print("COMPREHENSIVE COMPARISON RESULTS")
print("="*70)

sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

print(f"{'Method':<15} {'Features':<8} {'Accuracy':<10} {'Rank'}")
print("-" * 45)

for rank, (method, result) in enumerate(sorted_results, 1):
    print(f"{method:<15} {result['n_features']:<8} {result['accuracy']:<10.4f} {rank}")

print(f"GA+SFS features: {sorted(ga_sfs_final_mask)}")
print(f"Pure SFS features: {sorted(sfs_features)}")

# Performance improvement analysis
print(f"\n" + "="*50)
print("PERFORMANCE IMPROVEMENT ANALYSIS")
print("="*50)

baseline_acc = all_features_score
best_method = sorted_results[0][0]
best_acc = sorted_results[0][1]['accuracy']

print(f"Baseline (All Features): {baseline_acc:.4f}")
print(f"Best Method ({best_method}): {best_acc:.4f}")
print(f"Improvement: {(best_acc - baseline_acc)*100:+.2f} percentage points")

# GA+SFS specific analysis
ga_sfs_improvement = (ga_sfs_score - ga_only_score) * 100
print(f"\nGA+SFS vs GA Only:")
print(f"GA Only: {ga_only_score:.4f}")
print(f"GA+SFS:  {ga_sfs_score:.4f}")
print(f"SFS Refinement Improvement: {ga_sfs_improvement:+.2f} percentage points")
