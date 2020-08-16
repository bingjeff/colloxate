from xml.etree import ElementTree

import pose


def get_link_map(joint_nodes, link_reference):
    link_map = {}
    for joint_node in joint_nodes:
        link_name = joint_node.find(link_reference).get('link')
        if link_name in link_map:
            link_map[link_name].append(joint_node)
        else:
            link_map[link_name] = [joint_node]
    return link_map


def get_joint_chains(parent_link_map, child_link, list_of_chains=None):
    old_chains = list_of_chains if list_of_chains else [[]]
    new_chains = []
    if child_link in parent_link_map:
        for joint_node in parent_link_map[child_link]:
            new_chains.extend(
                get_joint_chains(parent_link_map,
                                 joint_node.find('child').get('link'),
                                 [oc + [joint_node] for oc in old_chains]))
        return new_chains
    else:
        return old_chains


def get_all_chains(urdf_root):
    joint_nodes = urdf_root.findall('joint')

    child_link_map = get_link_map(joint_nodes, 'child')
    parent_link_map = get_link_map(joint_nodes, 'parent')

    parent_link_set = set([v for v in parent_link_map])
    child_link_set = set([v for v in child_link_map])

    root_links = list(parent_link_set.difference(child_link_set))
    joint_chains = []
    for root_link in root_links:
        for joint_chain in get_joint_chains(parent_link_map, root_link):
            joint_chains.append(joint_chain)

    return joint_chains


def get_chain(urdf_root, root_link_name, tip_link_name):
    joint_nodes = urdf_root.findall('joint')
    parent_link_map = get_link_map(joint_nodes, 'parent')
    joint_chains = get_joint_chains(parent_link_map, root_link_name)
    for chain in joint_chains:
        tip_joint = chain[-1]
        link_name = tip_joint.find('child').get('link')
        if link_name == tip_link_name:
            return chain
    return None


def parse_string_to_numeric_list(vec_string):
    return [float(x) for x in vec_string.split(' ') if x]


def read_all_chains_from_urdf(urdf_path):
    with open(urdf_path, 'r') as f:
        urdf_string = f.read()
    urdf_root = ElementTree.fromstring(urdf_string)
    return get_all_chains(urdf_root)


def read_chain_from_urdf(urdf_path, root_link_name, tip_link_name):
    with open(urdf_path, 'r') as f:
        urdf_string = f.read()
    urdf_root = ElementTree.fromstring(urdf_string)
    return get_chain(urdf_root, root_link_name, tip_link_name)


def make_rpy_xyz_pose(rpy, xyz):
    yaw = pose.make_pose(xyz, rpy[2], [0.0, 0.0, 1.0])
    pitch = pose.make_pose([0., 0., 0.], rpy[1], [0.0, 1.0, 0.0])
    roll = pose.make_pose([0., 0., 0.], rpy[0], [1.0, 0.0, 0.0])
    return pose.multiply(pose.multiply(yaw, pitch), roll)


def extract_origin_pose(joint_node):
    origin = joint_node.find('origin')
    if origin is not None:
        xyz = parse_string_to_numeric_list(origin.get('xyz'))
        rpy = parse_string_to_numeric_list(origin.get('rpy'))
        return make_rpy_xyz_pose(rpy, xyz)
    else:
        return pose.make_identity_pose()


def extract_axis_function(joint_node):
    xyz = parse_string_to_numeric_list(joint_node.find('axis').get('xyz'))
    return lambda rotation: pose.make_pose([0., 0., 0.], rotation, xyz)


def kinematic_chain_function(joint_values, joint_operations):
    tip_pose = pose.make_identity_pose()
    for v, op in zip(joint_values, joint_operations):
        tip_pose = pose.multiply(tip_pose, op(v))
    return tip_pose


def make_kinematic_chain_function(joint_chain):
    operations = []
    prev_pose = pose.make_identity_pose()
    for joint in joint_chain:
        prev_pose = pose.multiply(prev_pose, extract_origin_pose(joint))
        if joint.get('type') in ['revolute', 'continuous']:
            joint_axis = extract_axis_function(joint)
            operations.append(
                lambda x, _prev_pose=prev_pose, _joint_axis=joint_axis: pose.multiply(
                    _prev_pose, _joint_axis(x)))
            prev_pose = pose.make_identity_pose()

    return lambda v: kinematic_chain_function(v, operations)
