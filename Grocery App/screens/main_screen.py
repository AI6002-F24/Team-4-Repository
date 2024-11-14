from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create a layout to hold the buttons in the center
        main_layout = BoxLayout(orientation='vertical')

        # Create a centered grid layout for the buttons
        button_layout = GridLayout(cols=3, size_hint=(None, None), size=(600, 600), spacing=20)
        button_layout.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        # Create square buttons with size_hint set to None for fixed size
        open_camera_button = Button(text="Open Camera", size_hint=(None, None), size=(200, 200))
        inventory_button = Button(text="Inventory", size_hint=(None, None), size=(200, 200))
        generate_recipe_button = Button(text="Generate Recipe", size_hint=(None, None), size=(200, 200))

        # Bind button events to functions
        open_camera_button.bind(on_release=self.change_screen_open_camera)
        inventory_button.bind(on_release=self.change_screen_inventory)
        generate_recipe_button.bind(on_release=self.change_screen_generate_recipe)

        # Add buttons to the layout
        button_layout.add_widget(open_camera_button)
        button_layout.add_widget(inventory_button)
        button_layout.add_widget(generate_recipe_button)

        # Add the button layout to the main layout
        main_layout.add_widget(button_layout)

        # Add the main layout to the screen
        self.add_widget(main_layout)

    def change_screen_open_camera(self, *args):
        print("Button Pressed: Navigating to OpenCameraScreen")  # Debug print
        self.manager.current = "open_camera_screen"

    def change_screen_inventory(self, *args):
        print("Navigating to InventoryScreen")  # Debug print
        self.manager.current = "inventory_screen"

    def change_screen_generate_recipe(self, *args):
        print("Navigating to GenerateRecipeScreen")  # Debug print
        self.manager.current = "generate_recipe_screen"
