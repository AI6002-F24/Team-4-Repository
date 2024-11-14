from kivymd.uix.screen import MDScreen
from kivymd.uix.label import MDLabel
from kivy.uix.boxlayout import BoxLayout

class WelcomeScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create a BoxLayout to center the text, ensuring proper vertical alignment
        layout = BoxLayout(orientation="vertical", padding=[50, 50], spacing=20)
        layout.size_hint = (1, 1)  # Make layout fill the entire screen
        layout.pos_hint = {"center_x": 0.5, "center_y": 0.5}  # Center the layout on the screen

        # Welcome text
        welcome_label = MDLabel(
            text="Welcome to the App!",
            halign="center",  # Horizontally align to the center
            theme_text_color="Primary",
            font_style="H4"
        )

        # Tap to start text
        tap_to_start_label = MDLabel(
            text="Tap to start!",
            halign="center",  # Horizontally align to the center
            theme_text_color="Secondary",
            font_style="Body1"
        )

        # Add the labels to the layout
        layout.add_widget(welcome_label)
        layout.add_widget(tap_to_start_label)

        # Add layout to the screen
        self.add_widget(layout)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.manager.current = "main_screen"
