import os
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
import requests

class GenerateRecipeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "generate_recipe_screen"

        # Scrollable main layout
        scroll_view = ScrollView(size_hint=(1, 1))
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint_y=None)
        main_layout.bind(minimum_height=main_layout.setter('height'))

        # Layout to hold the label and scrollable box side by side
        self.ingredient_box_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=60)
        ingredients_label = Label(text="Available Ingredients:", size_hint=(None, None), size=(180, 40), color=(0, 0, 0, 1))
        self.ingredient_box_layout.add_widget(ingredients_label)

        # Scrollable widget for item names
        scroll_view_items = ScrollView(size_hint=(1, None), height=70)
        self.item_name_layout = GridLayout(cols=len(self.load_item_names()) if self.load_item_names() else 1, size_hint_x=None, spacing=7)
        self.item_name_layout.bind(minimum_width=self.item_name_layout.setter('width'))

        scroll_view_items.add_widget(self.item_name_layout)
        self.ingredient_box_layout.add_widget(scroll_view_items)
        main_layout.add_widget(self.ingredient_box_layout)

        # Ingredients input field with label
        ingredients_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=40)
        ingredients_label = Label(text="Ingredients:", size_hint=(None, None), size=(100, 40), color=(0, 0, 0, 1))
        self.ingredients_input = TextInput(size_hint=(1, None), height=40, multiline=False)
        ingredients_layout.add_widget(ingredients_label)
        ingredients_layout.add_widget(self.ingredients_input)
        main_layout.add_widget(ingredients_layout)

        # Cooking time input field with label
        time_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=40)
        time_label = Label(text="Cooking Time:", size_hint=(None, None), size=(100, 40), color=(0, 0, 0, 1))
        self.time_input = TextInput(size_hint=(1, None), height=40, multiline=False)
        time_layout.add_widget(time_label)
        time_layout.add_widget(self.time_input)
        main_layout.add_widget(time_layout)

        # Directions display text box with label
        # Create a label for directions
        directions_label = Label(text="Directions for Recipe:", size_hint=(None, None), size=(180, 40), color=(0, 0, 0, 1))
        main_layout.add_widget(directions_label)

        # Create a ScrollView for the directions input box
        # Set the ScrollView to take up available space but with some height restriction
        directions_scroll_view = ScrollView(size_hint=(1, None), height=400)  # This can be adjusted for your UI

        # Create the directions input field inside the ScrollView
        self.directions_input = TextInput(size_hint=(1, None), height=200, readonly=True, multiline=True)  # Allow multiline input
        directions_scroll_view.add_widget(self.directions_input)

        # Add the ScrollView with TextInput to the main layout
        main_layout.add_widget(directions_scroll_view)

        # Generate Recipe button
        generate_button = Button(text="Generate Recipe", size_hint=(1, None), height=40)
        generate_button.bind(on_press=self.send_data_to_server)
        main_layout.add_widget(generate_button)

        back_button = Button(text="Back", size_hint=(1, None), height=40)
        back_button.bind(on_press=self.go_back)
        main_layout.add_widget(back_button)
        
        # Add main layout to the scroll view
        scroll_view.add_widget(main_layout)
        self.add_widget(scroll_view)

    def on_enter(self):
        """Called when the screen is entered. Reload ingredients."""
        self.reload_ingredients()

    def load_item_names(self):
        """Load item names from the JSON file."""
        json_file_path = os.path.join(os.path.dirname(__file__), 'ingredients', 'output.json')
        item_names = []
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                for item in data:
                    item_names.append(item.get('Item Name', 'Unknown'))
        return item_names

    def send_data_to_server(self, instance):
        # Collect the data from the input fields
        ingredients = self.ingredients_input.text.split(',')
        cooking_time = self.time_input.text
        
        # Create the data dictionary to send
        data = {
            'ingredients': [ingredient.strip() for ingredient in ingredients],
            'cooking_time': cooking_time
        }
        
        # Send the data to the server using a POST request
        try:
            response = requests.post('http://192.168.2.117:5555/receive_data', json=data)
            if response.status_code == 200:
                result = response.json()
                self.directions_input.text = result.get('recipe_directions', 'No recipe directions found.')
            else:
                self.directions_input.text = f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            self.directions_input.text = f"Failed to connect to server: {e}"

    def reload_ingredients(self):
        """Force reload item names from the JSON file."""
        # Trigger reloading of item names and update the display
        self.item_name_layout.clear_widgets()
        item_names = self.load_item_names()
        if item_names:
            for name in item_names:
                label = Label(text=name, size_hint=(None, None), size=(120, 40), halign='center', color=(0, 0, 0, 1))
                self.item_name_layout.add_widget(label)
        else:
            label = Label(text="No items found", size_hint=(None, None), size=(120, 40), halign='center', color=(0, 0, 0, 1))
            self.item_name_layout.add_widget(label)

    def go_back(self, instance):
        """Navigate back to the main screen."""
        self.manager.current = 'main_screen'