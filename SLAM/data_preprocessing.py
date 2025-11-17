"""
Data preprocessing module for SLAM dataset.
Parses data the same way as baseline.py and prepares LLM input format.
"""

from collections import defaultdict
from io import open


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

