"""Streamlit utility functions."""

import streamlit as st


_STYLE_TAG = "<style> {} </style>"


def set_hr_vertical_margin(em: float = 1) -> None:
    """Reduces the vertical margin of the horizontal rule."""
    inner = f"hr {{margin: {em}em 0px;}}"
    st.markdown(_STYLE_TAG.format(inner), unsafe_allow_html=True)


def hide_altair_fullscreen_button() -> None:
    """Hides the fullscreen button."""
    inner = "button[title='View fullscreen'] {display: none;}"
    st.markdown(_STYLE_TAG.format(inner), unsafe_allow_html=True)


def set_main_block_marin() -> None:
    """"""
    inner = ".stMainBlockContainer {padding-top: 4rem;}"
    st.markdown(_STYLE_TAG.format(inner), unsafe_allow_html=True)


def set_sidebar_header_padding() -> None:
    """Hides the sidebar header."""
    inner = "div[data-testid='stSidebarHeader'] {padding-bottom: 0.5rem;}"
    st.markdown(_STYLE_TAG.format(inner), unsafe_allow_html=True)


def set_dialog_width() -> None:
    """Sets the dialog width."""
    inner = "div[data-testid='stDialog'] > div:first-child > div:first-child {width: 80% !important;}"
    st.markdown(_STYLE_TAG.format(inner), unsafe_allow_html=True)


def set_default_styles() -> None:
    """Sets the default styles for the app."""
    set_hr_vertical_margin()
    hide_altair_fullscreen_button()
    set_dialog_width()
