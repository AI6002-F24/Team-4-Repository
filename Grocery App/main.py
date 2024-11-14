from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import ScreenManager

# Import screen files
from screens.welcome_screen import WelcomeScreen
from screens.main_screen import MainScreen  
from screens.open_camera_screen import OpenCameraScreen
from screens.inventory_screen import InventoryScreen
from screens.generate_recipe_screen import GenerateRecipeScreen
from screens.edit_item_screen import EditItemScreen

class MyApp(MDApp):
    def build(self):
        # Create ScreenManager instance
        screen_manager = ScreenManager()

        # Manually add screens to the screen manager
        screen_manager.add_widget(WelcomeScreen(name="welcome_screen"))
        screen_manager.add_widget(MainScreen(name="main_screen"))
        screen_manager.add_widget(OpenCameraScreen(name="open_camera_screen"))
        screen_manager.add_widget(InventoryScreen(name="inventory_screen"))
        screen_manager.add_widget(GenerateRecipeScreen(name="generate_recipe_screen"))
        screen_manager.add_widget(EditItemScreen(name="edit_item_screen"))

        return screen_manager

if __name__ == "__main__":
    MyApp().run()
