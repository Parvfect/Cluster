
import numpy as np

class Node:

    def __init__(self, no_connections, identifier, value=0, links=None):
        self.value = value
        self.identifier = identifier
        if links:
            self.links = links
        else:
            self.links = [None for i in range(no_connections)]

    def change_value(self, new_value):
        self.value = new_value    

    def add_link(self, node):
        """ Adds a link to the node. Throws an error if the node is full """
        for (i,j) in enumerate(self.links):
            if not j:
                self.links[i] = node
                break   
        return self.links
    
    def get_links(self):
        return self.links

    def replace_link(self, node, index):
        """ Replaces a link with another link """
        self.links[index] = node
        return self.links
    
    def get_value(self):
        return self.value
    
class CheckNode(Node):

    def __init__(self, dc, identifier, links = None):
        super().__init__(dc, identifier, links = links)

    def get_total_symbol_possibilities(self):
        sum=0
        for i in self.links:
            sum+=i.total_symbol_possibilities
        return sum
        #return sum([i.total_symbol_possibilities for i in self.links])
    
class VariableNode(Node):
    def __init__(self, dv, identifier, value=None):
        super().__init__(dv, identifier, value)
        if value is not None:
            self.update_symbol_possibilities(len(value))

    def get_total_symbol_possibilities(self):
        return self.total_symbol_possibilities
    
    def update_symbol_possibilities(self, new_poss):
        self.total_symbol_possibilities = new_poss
    
    def change_value(self, new_value):
        super().change_value(new_value)
        self.update_symbol_possibilities(len(new_value))

class Link(Node):
    def __init__(self, cn, vn, value):
        self.cn = cn
        self.vn = vn
        self.value = value

class TreeNode:
    def __init__(self, cn_node, left, right):
        self.cn = cn_node
        self.left = left
        self.right = right
    
    def get_cn_node(self):
        return self.cn
    
    def get_cn_value(self):
        return self.cn.get_total_symbol_possibilities()

    def get_cn_identifier(self):
        return self.cn.identifier

    def get_left(self):
        return self.left
    
    def get_right(self):
        return self.right
    
    def add_right(self, cn_node):
        self.right = cn_node
    
    def add_left(self, cn_node):
        self.left = cn_node

    def remove_left(self):
        left = self.get_left().get_cn_node()
        self.left = None
        return left

    def remove_right(self):
        right = self.get_right().get_cn_node()
        self.right = None
        return right
    
    def has_neighbours(self):
        return self.get_right() or self.get_left()


class ValueTree:
    """Value Tree to represent sorted CN List. Using CN Nodes as the nodes themselves, so accessing value is through cn.val, and identifier is cn.identifier. """

    def __init__(self, nodes=None):
        self.root = None
        self.length = 0
        if nodes:
            for i in nodes:
                self.add_node(i)
    
    def get_length(self):
        return self.length
    
    def is_empty(self):
        return self.root==None or self.length==0
    
    def add_node(self, cn_node):

        self.length += 1

        if self.root == None:
            self.root = TreeNode(cn_node, None, None)
            return 
        
        ptr = self.root
        value = cn_node.get_total_symbol_possibilities()

        while True:
            ptr_symbol_poss = ptr.get_cn_value()
            if value >= ptr_symbol_poss:
                if ptr.get_right() is not None:
                    ptr = ptr.right
                else:
                    ptr.add_right(TreeNode(cn_node, None, None))
                    return
                
            if value < ptr_symbol_poss:
                if ptr.get_left() is not None:
                    ptr = ptr.left
                else:
                    ptr.add_left(TreeNode(cn_node, None, None))
                    return
        
    def remove_smallest_node(self):
        """Using to Traverse tree for bottom to top for CC Decoder """

        self.length-=1
        if self.root == None:
            print("Tree is empty")
            return None
    
        ptr1 = self.root
        ptr2 = self.root

        if ptr1.get_left() is None:
            cn_node = self.root.get_cn_node()
            self.root = ptr1.get_right()
            return cn_node

        while ptr1.has_neighbours():
            ptr2 = ptr1
            if ptr1.get_left():
                ptr1 = ptr1.get_left()
            else:
                ptr1 = ptr1.get_right()
        
        if ptr2.get_left():
            return ptr2.remove_left()
        
        return ptr2.remove_right()
    
    def find_vn_node(self, cn_node):
        """Uses both value and identifier to distinguish """
        return


def testing():
    k = 1000
    n = 400
    t = np.arange(1, n)

    vns = []
    # Creating Variable Nodes
    for i in range(n):
        vns.append(VariableNode(3, i, np.random.rand(np.random.choice(np.arange(67)))))
        print(i)

    cns = []

    for i in range(k):
        cns.append(CheckNode(9, i, links=[np.random.choice(vns) for i in range(9)]))
        print(i)

    tree = ValueTree(cns)
    print()

    for i in range(k):
        print(tree.remove_smallest_node().total_symbol_possibilites())
        