from radiology_trainer.ui import APP_CSS, create_app, create_theme


if __name__ == "__main__":
    create_app().launch(css=APP_CSS, theme=create_theme())
