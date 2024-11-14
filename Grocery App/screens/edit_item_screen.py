from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
import json
import os

class EditItemScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output = []  # Store the items from OCR data
        self.index = 1  # Current item index
        
        # Define the layout for the screen
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Define input fields with labels and set label text properties
        self.item_name_label = Label(text="Item Name:", color=(0, 0, 0, 1), bold=True, font_size='20sp')  # black color, bold, larger font size
        self.item_name_input = TextInput(hint_text="Enter Item Name", multiline=False)
        
        self.item_price_label = Label(text="Item Price:", color=(0, 0, 0, 1), bold=True, font_size='20sp')  # black color, bold, larger font size
        self.item_price_input = TextInput(hint_text="Enter Item Price", multiline=False)
        
        self.item_quantity_label = Label(text="Item Quantity:", color=(0, 0, 0, 1), bold=True, font_size='20sp')  # black color, bold, larger font size
        self.item_quantity_input = TextInput(hint_text="Enter Item Quantity", multiline=False)
        
        # Create a BoxLayout for each label and its corresponding input field to control alignment
        name_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        name_layout.add_widget(self.item_name_label)
        name_layout.add_widget(self.item_name_input)

        price_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        price_layout.add_widget(self.item_price_label)
        price_layout.add_widget(self.item_price_input)

        quantity_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        quantity_layout.add_widget(self.item_quantity_label)
        quantity_layout.add_widget(self.item_quantity_input)
        
        # Define buttons
        self.next_button = Button(text="Next", on_press=self.save_and_next)
        self.previous_button = Button(text="Previous", on_press=self.show_previous_item)
        self.add_button = Button(text="Add Item", on_press=self.add_new_item, disabled=True)
        self.done_button = Button(text="Done", on_press=self.save_and_done, disabled=True)
        self.remove_button = Button(text="Remove", on_press=self.remove_item, disabled=False)  # Add remove button

        # BoxLayout to align buttons side by side
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50, spacing=10)
        button_layout.add_widget(self.previous_button)
        button_layout.add_widget(self.next_button)
        button_layout.add_widget(self.add_button)
        button_layout.add_widget(self.done_button)
        button_layout.add_widget(self.remove_button)  # Add remove button to the layout

        # Add widgets to layout
        self.layout.add_widget(name_layout)
        self.layout.add_widget(price_layout)
        self.layout.add_widget(quantity_layout)

        # Add button layout
        self.layout.add_widget(button_layout)

        # Add layout to screen
        self.add_widget(self.layout)

    def load_data(self, json_data):
        """Load data from the OCR result (JSON)"""
        self.output = json_data  # Assign the OCR data to output
        self.show_item(self.index)

    def show_item(self, index):
        """Display the current item details in the inputs"""
        if 0 <= index < len(self.output):
            item = self.output[index]
            self.item_name_input.text = item['Item Name']
            self.item_price_input.text = str(item['Item Price'])
            self.item_quantity_input.text = str(item['Item Quantity'])
            # Enable the "Next" button and disable "Previous" if we're at the first item
            self.previous_button.disabled = (self.index == 0)
            # If we're at the last item, disable "Next" button and enable "Add Item" and "Done"
            self.next_button.disabled = (self.index == len(self.output))
            if self.index == len(self.output) - 1:
                self.add_button.disabled = False  # Enable "Add Item"
                self.done_button.disabled = False  # Enable "Done"
            else:
                self.add_button.disabled = True  # Keep "Add Item" disabled until the last item
                self.done_button.disabled = True  # Keep "Done" disabled until the last item
        else:
            # Reset fields if out of range
            self.item_name_input.text = ''
            self.item_price_input.text = ''
            self.item_quantity_input.text = ''

    def save_and_next(self, instance):
        """Save current item and move to next item"""
        self.output[self.index] = {
            'Item Name': self.item_name_input.text,
            'Item Price': self.item_price_input.text,
            'Item Quantity': self.item_quantity_input.text
        }
        self.index += 1
        if self.index >= len(self.output):
            self.index = len(self.output) - 1
        self.show_item(self.index)

    def show_previous_item(self, instance):
        """Go back to previous item"""
        self.index -= 1
        if self.index < 0:
            self.index = 0
        self.show_item(self.index)

    def add_new_item(self, instance):
        """Prepare to add a new item"""
        # Before adding the new empty item, save the current item if it has data
        if self.item_name_input.text and self.item_price_input.text and self.item_quantity_input.text:
            self.output[self.index] = {
                'Item Name': self.item_name_input.text,
                'Item Price': self.item_price_input.text,
                'Item Quantity': self.item_quantity_input.text
            }

        # Clear the input fields for new item
        self.item_name_input.text = ''
        self.item_price_input.text = ''
        self.item_quantity_input.text = ''

        # Add the new empty item to output
        self.output.append({
            'Item Name': '',
            'Item Price': '',
            'Item Quantity': ''
        })

        # Set the index to the new item and show it
        self.index = len(self.output) - 1  # Start editing the newly added item
        self.show_item(self.index)

    def remove_item(self, instance):
        """Remove all text from the current item's inputs"""
        self.item_name_input.text = ''
        self.item_price_input.text = ''
        self.item_quantity_input.text = ''

    def save_and_done(self, instance):
        """Save the data and navigate back to main screen"""
        # Before saving, make sure the current item is saved if it has data
        if self.item_name_input.text and self.item_price_input.text and self.item_quantity_input.text:
            self.output[self.index] = {
                'Item Name': self.item_name_input.text,
                'Item Price': self.item_price_input.text,
                'Item Quantity': self.item_quantity_input.text
            }

        # Get the path of the current directory where the script is located
        current_dir = os.path.dirname(__file__)

        # Create the full path for the 'output.json' file within the 'ingredients' folder
        json_file_path = os.path.join(current_dir, 'ingredients', 'output.json')

        # Save the JSON data to the file
        with open(json_file_path, 'w') as json_file:
            json.dump(self.output, json_file)

        # Return to the main screen
        self.manager.current = 'main_screen'
