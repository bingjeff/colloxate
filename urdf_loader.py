from xml.etree import ElementTree as element_tree

import pose


def GetLinkMap(joint_nodes, link_reference):
    link_map = {}
    for joint_node in joint_nodes:
        link_name = joint_node.find(link_reference).get('link')
        if link_name in link_map:
            link_map[link_name].append(joint_node)
        else:
            link_map[link_name] = [joint_node]
    return link_map


def GetJointChains(parent_link_map, child_link, list_of_chains=None):
    old_chains = list_of_chains if list_of_chains else [[]]
    new_chains = []
    if child_link in parent_link_map:
        for joint_node in parent_link_map[child_link]:
            new_chains.extend(
                GetJointChains(parent_link_map,
                               joint_node.find('child').get('link'),
                               [oc + [joint_node] for oc in old_chains]))
        return new_chains
    else:
        return old_chains


def GetAllChains(urdf_root):
    joint_nodes = urdf_root.findall('joint')

    child_link_map = GetLinkMap(joint_nodes, 'child')
    parent_link_map = GetLinkMap(joint_nodes, 'parent')

    parent_link_set = set([v for v in parent_link_map])
    child_link_set = set([v for v in child_link_map])

    root_links = list(parent_link_set.difference(child_link_set))
    joint_chains = []
    for root_link in root_links:
        for joint_chain in GetJointChains(parent_link_map, root_link):
            joint_chains.append(joint_chain)

    return joint_chains


def GetChain(urdf_root, root_link_name, tip_link_name):
    joint_nodes = urdf_root.findall('joint')
    parent_link_map = GetLinkMap(joint_nodes, 'parent')
    joint_chains = GetJointChains(parent_link_map, root_link_name)
    for chain in joint_chains:
        tip_joint = chain[-1]
        link_name = tip_joint.find('child').get('link')
        if link_name == tip_link_name:
            return chain
    return None


def ParseStringToNumericList(vec_string):
    return [float(x) for x in vec_string.split(' ') if x]


def ReadAllChainsFromUrdf(urdf_path):
    with open(urdf_path, 'r') as f:
        urdf_string = f.read()
    urdf_root = element_tree.fromstring(urdf_string)
    return GetAllChains(urdf_root)


def ReadChainFromUrdf(urdf_path, root_link_name, tip_link_name):
    with open(urdf_path, 'r') as f:
        urdf_string = f.read()
    urdf_root = element_tree.fromstring(urdf_string)
    return GetChain(urdf_root, root_link_name, tip_link_name)


def MakeRpyXyzPose(rpy, xyz):
    yaw = pose.MakePose(xyz, rpy[2], [0.0, 0.0, 1.0])
    pitch = pose.MakePose([0., 0., 0.], rpy[1], [0.0, 1.0, 0.0])
    roll = pose.MakePose([0., 0., 0.], rpy[0], [1.0, 0.0, 0.0])
    return pose.MultiplyPoses(pose.MultiplyPoses(yaw, pitch), roll)


def ExtractOriginPose(joint_node):
    origin = joint_node.find('origin')
    if origin is not None:
        xyz = ParseStringToNumericList(origin.get('xyz'))
        rpy = ParseStringToNumericList(origin.get('rpy'))
        return MakeRpyXyzPose(rpy, xyz)
    else:
        return pose.MakeIdentityPose()


def ExtractAxisFunction(joint_node):
    xyz = ParseStringToNumericList(joint_node.find('axis').get('xyz'))
    return lambda rotation: pose.MakePose([0., 0., 0.], rotation, xyz)


def KinematicChainFunction(joint_values, joint_operations):
    tip_pose = pose.MakeIdentityPose()
    for v, op in zip(joint_values, joint_operations):
        tip_pose = pose.MultiplyPoses(tip_pose, op(v))
    return tip_pose


def MakeKinematicChainFunction(joint_chain):
    operations = []
    prev_pose = pose.MakeIdentityPose()
    for joint in joint_chain:
        prev_pose = pose.MultiplyPoses(prev_pose, ExtractOriginPose(joint))
        if joint.get('type') in ['revolute', 'continuous']:
            joint_axis = ExtractAxisFunction(joint)
            operations.append(lambda x, prev_pose=prev_pose, joint_axis=joint_axis: pose.MultiplyPoses(
                prev_pose, joint_axis(x)))
            prev_pose = pose.MakeIdentityPose()

    return lambda v: KinematicChainFunction(v, operations)
