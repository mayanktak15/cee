#Implement a Linked List in Python Using OOP and Delete the Nth Node
#Create a Python program that implements a singly linked list using Object-Oriented Programming (OOP) principles. Your implementation should include the following: A Node class to represent each node in the list. A LinkedList class to manage the nodes, with methods to: Add a node to the end of the list Print the list Delete the nth node (where n is a 1-based index) Include exception handling to manage edge cases such as: Deleting a node from an empty list Deleting a node with an index out of range Test your implementation with at least one sample list.
# Implement a Linked List in Python Using OOP and Delete the Nth Node
# Create a Python program that implements a singly linked list using Object-Oriented Programming (OOP) principles. Your implementation should include the following: 
# - A Node class to represent each node in the list. 
# - A LinkedList class to manage the nodes, with methods to: 
#   - Add a node to the end of the list 
#   - Print the list 
#   - Delete the nth node (where n is a 1-based index) 
# Include exception handling to manage edge cases such as: 
#   - Deleting a node from an empty list 
#   - Deleting a node with an index out of range 
# Test your implementation with at least one sample list.

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        current = self.head
        if not current:
            print("List is empty.")
            return
        while current:
            print(current.data, end=" -> " if current.next else "\n")
            current = current.next

    def delete_nth_node(self, n):
        if not self.head:
            raise Exception("Cannot delete from an empty list.")
        if n <= 0:
            raise Exception("Index must be a positive integer.")
        if n == 1:
            self.head = self.head.next
            return
        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1
        if not current or not current.next:
            raise Exception("Index out of range.")
        current.next = current.next.next

# Sample test
if __name__ == "__main__":
    ll = LinkedList()
    ll.add_node(10)
    ll.add_node(20)
    ll.add_node(30)
    ll.add_node(40)
    print("Original list:")
    ll.print_list()

    try:
        ll.delete_nth_node(3)
        print("After deleting 3rd node:")
        ll.print_list()
        ll.delete_nth_node(1)
        print("After deleting 1st node:")
        ll.print_list()
        ll.delete_nth_node(10)  # Should raise exception
    except Exception as e:
        print("Exception:", e)