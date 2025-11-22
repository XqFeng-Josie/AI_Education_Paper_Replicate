"""
Data preprocessing module for SLAM dataset.
Parses data the same way as baseline.py and prepares LLM input format.
"""

from collections import defaultdict, Counter
from io import open
import statistics
import argparse
import os

def load_data(filename):
    """
    This method loads and returns the data in filename. If the data is labelled training data, it returns labels too.
    This is the same function as in baseline.py to ensure consistency.

    Parameters:
        filename: the location of the training or test data you want to load.

    Returns:
        data: a list of InstanceData objects from that data type and track.
        labels (optional): if you specified training data, a dict of instance_id:label pairs.
    """
    data = []
    training = False
    if filename.find('train') != -1:
        training = True

    if training:
        labels = dict()

    num_exercises = 0
    print('Loading instances...')
    instance_properties = dict()

    with open(filename, 'rt') as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                num_exercises += 1
                if num_exercises % 100000 == 0:
                    print('Loaded ' + str(len(data)) + ' instances across ' + str(num_exercises) + ' exercises...')
                instance_properties = dict()

            elif line[0] == '#':
                if 'prompt' in line:
                    instance_properties['prompt'] = line.split(':')[1]
                else:
                    list_of_exercise_parameters = line[2:].split()
                    for exercise_parameter in list_of_exercise_parameters:
                        [key, value] = exercise_parameter.split(':')
                        if key == 'countries':
                            value = value.split('|')
                        elif key == 'days':
                            value = float(value)
                        elif key == 'time':
                            if value == 'null':
                                value = None
                            else:
                                assert '.' not in value
                                value = int(value)
                        instance_properties[key] = value

            else:
                line = line.split()
                if training:
                    assert len(line) == 7
                else:
                    assert len(line) == 6
                assert len(line[0]) == 12

                instance_properties['instance_id'] = line[0]
                instance_properties['token'] = line[1]
                instance_properties['part_of_speech'] = line[2]

                instance_properties['morphological_features'] = dict()
                for l in line[3].split('|'):
                    [key, value] = l.split('=')
                    if key == 'Person':
                        value = int(value)
                    instance_properties['morphological_features'][key] = value

                instance_properties['dependency_label'] = line[4]
                instance_properties['dependency_edge_head'] = int(line[5])
                if training:
                    label = float(line[6])
                    labels[instance_properties['instance_id']] = label
                data.append(InstanceData(instance_properties=instance_properties))

        print('Done loading ' + str(len(data)) + ' instances across ' + str(num_exercises) +
              ' exercises.\n')

    if training:
        return data, labels
    else:
        return data


class InstanceData(object):
    """
    A bare-bones class to store the included properties of each instance.
    Same as in baseline.py to ensure consistency.
    """
    def __init__(self, instance_properties):
        self.instance_id = instance_properties['instance_id']
        self.token = instance_properties['token']
        self.part_of_speech = instance_properties['part_of_speech']
        self.morphological_features = instance_properties['morphological_features']
        self.dependency_label = instance_properties['dependency_label']
        self.dependency_edge_head = instance_properties['dependency_edge_head']

        self.exercise_index = int(self.instance_id[8:10])
        self.token_index = int(self.instance_id[10:12])
        self.exercise_id = self.instance_id[:10]

        self.user = instance_properties['user']
        self.countries = instance_properties['countries']
        self.days = instance_properties['days']
        self.client = instance_properties['client']
        self.session = instance_properties['session']
        self.format = instance_properties['format']
        self.time = instance_properties['time']
        self.prompt = instance_properties.get('prompt', None)

        self.session_id = self.instance_id[:8]

    def to_llm_input_text(self):
        """
        Build a compact input text that includes only the baseline features:
        User, format, token (lowercased), part_of_speech, morphological feature names (keys only), dependency_label.
        
        This matches the features used in baseline.py's to_features() method.
        
        Returns:
            A string representation of the features for LLM input.
        """
        # Get morphological feature keys only (as in baseline: morphological_feature:key)
        morph_keys = list(self.morphological_features.keys())
        morph_str = ','.join(morph_keys) if morph_keys else 'none'
        
        # Build compact text representation matching baseline features
        # Format: User format token(lowercased) POS morph_keys DepLabel
        text_parts = [
            f"User:{self.user}",
            f"Format:{self.format}",
            f"Token:{self.token.lower()}",
            f"POS:{self.part_of_speech}",
            f"Morph:{morph_str}",
            f"DepLabel:{self.dependency_label}"
        ]
        
        return " ".join(text_parts)


def statistics_data(data, labels=None, verbose=True):
    """
    Statistics and distribution analysis of the SLAM dataset.
    
    Parameters:
        data: a list of InstanceData objects
        labels (optional): a dict of instance_id:label pairs (for training data)
        verbose: if True, print detailed statistics
    
    Returns:
        stats: a dict containing all statistics
    """
    if not data:
        print("Warning: Empty data list")
        return {}
    
    stats = {}
    
    # Basic statistics
    num_instances = len(data)
    unique_users = set(inst.user for inst in data)
    unique_exercises = set(inst.exercise_id for inst in data)
    unique_sessions = set(inst.session_id for inst in data)
    
    stats['basic'] = {
        'num_instances': num_instances,
        'num_users': len(unique_users),
        'num_exercises': len(unique_exercises),
        'num_sessions': len(unique_sessions),
        'avg_instances_per_user': num_instances / len(unique_users) if unique_users else 0,
        'avg_instances_per_exercise': num_instances / len(unique_exercises) if unique_exercises else 0,
    }
    
    # Label distribution (if available)
    if labels is not None:
        label_values = list(labels.values())
        # Normalize all label values to float to avoid type mismatch issues
        # (e.g., 1 vs 1.0 being treated as different keys in Counter)
        normalized_labels = [float(v) for v in label_values]
        label_counter = Counter(normalized_labels)
        
        # Count correct (1.0) and incorrect (0.0) labels
        num_correct = label_counter.get(1.0, 0)
        num_incorrect = label_counter.get(0.0, 0)
        total_labeled = len(label_values)
        
        # Check for unexpected label values
        unexpected_labels = {k: v for k, v in label_counter.items() if k not in [0.0, 1.0]}
        
        stats['labels'] = {
            'total_labeled': total_labeled,
            'correct (1)': num_correct,
            'incorrect (0)': num_incorrect,
            'correct_ratio': num_correct / total_labeled if total_labeled > 0 else 0,
            'incorrect_ratio': num_incorrect / total_labeled if total_labeled > 0 else 0,
            'unexpected_labels': unexpected_labels if unexpected_labels else None,
        }
    
    # Format distribution
    format_counter = Counter(inst.format for inst in data)
    stats['format'] = dict(format_counter)
    
    # Client distribution
    client_counter = Counter(inst.client for inst in data)
    stats['client'] = dict(client_counter)
    
    # Session type distribution
    session_counter = Counter(inst.session for inst in data)
    stats['session_type'] = dict(session_counter)
    
    # POS distribution
    pos_counter = Counter(inst.part_of_speech for inst in data)
    stats['pos'] = dict(pos_counter.most_common(20))  # Top 20 POS tags
    
    # Dependency label distribution
    dep_counter = Counter(inst.dependency_label for inst in data)
    stats['dependency_label'] = dict(dep_counter.most_common(20))  # Top 20 dependency labels
    
    # Morphological features distribution
    morph_feature_counter = Counter()
    for inst in data:
        for morph_key in inst.morphological_features.keys():
            morph_feature_counter[morph_key] += 1
    stats['morphological_features'] = dict(morph_feature_counter)
    
    # Days distribution (learning days)
    days_list = [inst.days for inst in data if inst.days is not None]
    if days_list:
        stats['days'] = {
            'min': min(days_list),
            'max': max(days_list),
            'mean': statistics.mean(days_list),
            'median': statistics.median(days_list),
            'stdev': statistics.stdev(days_list) if len(days_list) > 1 else 0,
        }
    
    # Countries distribution
    country_counter = Counter()
    for inst in data:
        if inst.countries:
            for country in inst.countries:
                country_counter[country] += 1
    stats['countries'] = dict(country_counter.most_common(10))  # Top 10 countries
    
    # User activity distribution
    user_instance_count = Counter(inst.user for inst in data)
    user_counts = list(user_instance_count.values())
    if user_counts:
        stats['user_activity'] = {
            'min_instances_per_user': min(user_counts),
            'max_instances_per_user': max(user_counts),
            'mean_instances_per_user': statistics.mean(user_counts),
            'median_instances_per_user': statistics.median(user_counts),
        }
    
    # Exercise size distribution
    exercise_instance_count = Counter(inst.exercise_id for inst in data)
    exercise_counts = list(exercise_instance_count.values())
    if exercise_counts:
        stats['exercise_size'] = {
            'min_instances_per_exercise': min(exercise_counts),
            'max_instances_per_exercise': max(exercise_counts),
            'mean_instances_per_exercise': statistics.mean(exercise_counts),
            'median_instances_per_exercise': statistics.median(exercise_counts),
        }
    
    # Print statistics if verbose
    if verbose:
        print("\n" + "=" * 80)
        print("DATA STATISTICS")
        print("=" * 80)
        
        print("\n[Basic Statistics]")
        print(f"  Total instances: {stats['basic']['num_instances']:,}")
        print(f"  Unique users: {stats['basic']['num_users']:,}")
        print(f"  Unique exercises: {stats['basic']['num_exercises']:,}")
        print(f"  Unique sessions: {stats['basic']['num_sessions']:,}")
        print(f"  Avg instances per user: {stats['basic']['avg_instances_per_user']:.2f}")
        print(f"  Avg instances per exercise: {stats['basic']['avg_instances_per_exercise']:.2f}")
        
        if labels is not None:
            print("\n[Label Distribution]")
            print(f"  Total labeled instances: {stats['labels']['total_labeled']:,}")
            print(f"  Correct (1): {stats['labels']['correct (1)']:,} ({stats['labels']['correct_ratio']*100:.2f}%)")
            print(f"  Incorrect (0): {stats['labels']['incorrect (0)']:,} ({stats['labels']['incorrect_ratio']*100:.2f}%)")
            
            # Warn about unexpected label values
            if stats['labels'].get('unexpected_labels'):
                unexpected_count = sum(stats['labels']['unexpected_labels'].values())
                print(f"  WARNING: Found {unexpected_count:,} instances with unexpected label values:")
                for label_val, count in sorted(stats['labels']['unexpected_labels'].items()):
                    print(f"    Label {label_val}: {count:,} instances")
            
            # Verify counts add up correctly
            total_counted = stats['labels']['correct (1)'] + stats['labels']['incorrect (0)']
            if total_counted != stats['labels']['total_labeled']:
                missing = stats['labels']['total_labeled'] - total_counted
                print(f"  WARNING: Label count mismatch! Total={stats['labels']['total_labeled']:,}, "
                      f"Counted={total_counted:,}, Missing={missing:,}")
        
        print("\n[Format Distribution]")
        for fmt, count in sorted(stats['format'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {fmt}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        print("\n[Client Distribution]")
        for client, count in sorted(stats['client'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {client}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        print("\n[Session Type Distribution]")
        for session, count in sorted(stats['session_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {session}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        print("\n[Top 10 POS Tags]")
        for pos, count in list(stats['pos'].items())[:10]:
            print(f"  {pos}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        print("\n[Top 10 Dependency Labels]")
        for dep, count in list(stats['dependency_label'].items())[:10]:
            print(f"  {dep}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        if stats.get('morphological_features'):
            print("\n[Morphological Features]")
            for morph, count in sorted(stats['morphological_features'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {morph}: {count:,} ({count/stats['basic']['num_instances']*100:.2f}%)")
        
        if stats.get('days'):
            print("\n[Learning Days Distribution]")
            print(f"  Min: {stats['days']['min']:.1f}")
            print(f"  Max: {stats['days']['max']:.1f}")
            print(f"  Mean: {stats['days']['mean']:.2f}")
            print(f"  Median: {stats['days']['median']:.2f}")
            print(f"  Std Dev: {stats['days']['stdev']:.2f}")
        
        if stats.get('countries'):
            print("\n[Top 10 Countries]")
            for country, count in list(stats['countries'].items())[:10]:
                print(f"  {country}: {count:,}")
        
        if stats.get('user_activity'):
            print("\n[User Activity]")
            print(f"  Min instances per user: {stats['user_activity']['min_instances_per_user']}")
            print(f"  Max instances per user: {stats['user_activity']['max_instances_per_user']}")
            print(f"  Mean instances per user: {stats['user_activity']['mean_instances_per_user']:.2f}")
            print(f"  Median instances per user: {stats['user_activity']['median_instances_per_user']:.2f}")
        
        if stats.get('exercise_size'):
            print("\n[Exercise Size]")
            print(f"  Min instances per exercise: {stats['exercise_size']['min_instances_per_exercise']}")
            print(f"  Max instances per exercise: {stats['exercise_size']['max_instances_per_exercise']}")
            print(f"  Mean instances per exercise: {stats['exercise_size']['mean_instances_per_exercise']:.2f}")
            print(f"  Median instances per exercise: {stats['exercise_size']['median_instances_per_exercise']:.2f}")
        
        print("\n" + "=" * 80)
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Analyze SLAM dataset statistics')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset (e.g., data_en_es)')
    parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'], required=True,
                        help='Data split to analyze')
    parser.add_argument('--quiet', action='store_true',
                        help='Only return stats dict, do not print')
    
    args = parser.parse_args()
    
    # Construct filename
    track = os.path.basename(args.data_dir.rstrip('/'))
    track = track.replace('data_', '')
    filename = os.path.join(args.data_dir, f"{track}.slam.20190204.{args.split}")
    
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return
    
    print(f"Loading data from: {filename}")
    
    # Load data
    if args.split == 'train':
        data, labels = load_data(filename)
    else:
        data = load_data(filename)
        labels = None
    
    # Calculate statistics
    stats = statistics_data(data, labels=labels, verbose=not args.quiet)
    
    # You can also access stats programmatically
    if not args.quiet:
        print("\n[Accessing stats programmatically]")
        print(f"  Total instances: {stats['basic']['num_instances']}")
        if labels is not None:
            print(f"  Correct ratio: {stats['labels']['correct_ratio']:.4f}")
        print(f"  Most common format: {max(stats['format'].items(), key=lambda x: x[1])[0]}")


if __name__ == '__main__':
    main()


