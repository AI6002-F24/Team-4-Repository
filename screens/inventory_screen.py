import os
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class InventoryScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "inventory_screen"

        # Main layout to hold the table (size_hint to ensure full-screen layout)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint=(1, 1))

        # ScrollView to enable scrolling if there are many items (fills the screen)
        scroll_view = ScrollView(size_hint=(1, 1))

        # GridLayout to act as the table (expandable height and row padding)
        self.table_layout = GridLayout(cols=3, spacing=10, size_hint_y=None, padding=[10, 10])
        self.table_layout.bind(minimum_height=self.table_layout.setter('height'))

        # Add headers to the table
        header_names = ["Item Name", "Item Price", "Item Quantity"]
        for header in header_names:
            header_label = Label(text=header, bold=True, color=(0, 0, 0, 1),
                                 size_hint_y=None, height=30)
            self.table_layout.add_widget(header_label)

        # Add the GridLayout to the ScrollView
        scroll_view.add_widget(self.table_layout)
        layout.add_widget(scroll_view)

        # Add the "Back" button at the bottom
        back_button = Button(text="Back", size_hint=(None, None), size=(100, 50), pos_hint={'left': 1, 'bottom': 0})
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)

        # Add the BoxLayout to the screen, ensuring it covers the full screen
        self.add_widget(layout)

    def on_enter(self):
        """Called every time the screen is entered to reload data"""
        self.load_inventory_data()

    def load_inventory_data(self):
        """Load and display data from output.json"""
        self.table_layout.clear_widgets()  # Clear current widgets (to refresh)
        
        json_file_path = os.path.join(os.path.dirname(__file__), 'ingredients', 'output.json')
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
            
            # Add headers to the table
            header_names = ["Item Name", "Item Price", "Item Quantity"]
            for header in header_names:
                header_label = Label(text=header, bold=True, color=(0, 0, 0, 1),
                                     size_hint_y=None, height=30)
                self.table_layout.add_widget(header_label)
            
            # Populate the table with data from the JSON file
            for item in data:
                item_name = Label(text=item.get('Item Name', ''), color=(0, 0, 0, 1),
                                  size_hint_y=None, height=30, text_size=(None, None), halign='center', valign='middle')
                item_price = Label(text=str(item.get('Item Price', '')), color=(0, 0, 0, 1),
                                   size_hint_y=None, height=30, text_size=(None, None), halign='center', valign='middle')
                item_quantity = Label(text=str(item.get('Item Quantity', '')), color=(0, 0, 0, 1),
                                      size_hint_y=None, height=30, text_size=(None, None), halign='center', valign='middle')
                self.table_layout.add_widget(item_name)
                self.table_layout.add_widget(item_price)
                self.table_layout.add_widget(item_quantity)
        else:
            empty_label = Label(text="No inventory data found.", color=(0, 0, 0, 1),
                                size_hint_y=None, height=30)
            self.table_layout.add_widget(empty_label)

    def go_back(self, instance):
        # Go back to the main screen
        self.manager.current = 'main_screen'
