import re
import numpy as np
import pandas as pd
import colour
import streamlit as st
from scipy.stats import norm
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class ExcitationData:
    """Data class to store excitation data."""
    energy_ev: float
    energy_cm: float
    wavelength_nm: float
    fosc: float
    D2: float
    DX: float
    DY: float
    DZ: float

    def validate(self) -> bool:
        """Validate the excitation data fields."""
        try:
            return (
                300 <= self.wavelength_nm <= 900
                and self.energy_ev > 0
                and self.energy_cm > 0
                and self.fosc >= 0
            )
        except (TypeError, ValueError):
            return False

class SpectrumCalculator:
    """Handle spectrum calculations and conversions."""
    def __init__(self, sigma: float = 20 / 2.355):
        """
        Initialize calculator with Gaussian width parameter.
        Args:
            sigma: Standard deviation for Gaussian (default: FWHM of 20 nm)
        """
        self.sigma = sigma
        self.shape = colour.SpectralShape(360, 830, 1)
        self.wavelengths = np.arange(360, 831, 1)
        self.cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        self.illuminant = colour.SDS_ILLUMINANTS["D65"]
        self.reference_white = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D65"
        ]

    @staticmethod
    def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
        """Generate Gaussian function for absorption band simulation."""
        return amplitude * norm.pdf(x, mu, sigma)

    def calculate_spectral_distribution(self, df: pd.DataFrame) -> Optional[colour.SpectralDistribution]:
        """Calculate combined spectral distribution from excitation data."""
        if df.empty:
            raise ValueError("Empty DataFrame. Cannot calculate spectral distribution.")

        combined_values = np.zeros_like(self.wavelengths, dtype=float)
        for _, row in df.iterrows():
            try:
                combined_values += self.gaussian(
                    self.wavelengths, row["wavelength_nm"], self.sigma, row["fosc"]
                )
            except KeyError:
                continue

        if not np.any(combined_values):
            raise ValueError("Failed to compute valid spectral distribution.")

        combined_values /= np.max(combined_values)
        return colour.SpectralDistribution(
            dict(zip(self.wavelengths, combined_values)), self.shape
        )

    def calculate_lab(self, spectral_dist: colour.SpectralDistribution) -> np.ndarray:
        """
        Calculate LAB values from spectral distribution with normalization.
        """
        # Convert spectral distribution to XYZ
        XYZ = colour.sd_to_XYZ(
            spectral_dist, cmfs=self.cmfs, illuminant=self.illuminant
        )
        logging.info(f"Raw XYZ values: {XYZ}")

        # Normalize XYZ values to reference white (von Kries chromatic adaptation)
        XYZ_normalized = XYZ / colour.sd_to_XYZ(
            self.illuminant, cmfs=self.cmfs, illuminant=self.illuminant
        )
        logging.info(f"Normalized XYZ values: {XYZ_normalized}")

        # Convert normalized XYZ to Lab
        Lab = colour.XYZ_to_Lab(XYZ_normalized, self.reference_white)
        logging.info(f"Unclipped Lab values: {Lab}")

        # Clip Lab values for validity
        L_MIN, L_MAX = 0, 100
        AB_MIN, AB_MAX = -128, 128
        Lab[0] = np.clip(Lab[0], L_MIN, L_MAX)  # L* ranges from 0 to 100
        Lab[1:] = np.clip(Lab[1:], AB_MIN, AB_MAX)  # a* and b* typically range from -128 to 128
        logging.info(f"Clipped Lab values: {Lab}")

        return Lab

class ExcitationDataParser:
    """Parse and validate TD-DFT excitation data."""
    @staticmethod
    def parse_data(data: str) -> Optional[pd.DataFrame]:
        """Improved TD-DFT data parsing."""
        try:
            lines = data.splitlines()
            start_index = next(
                i for i, line in enumerate(lines) if "ABSORPTION SPECTRUM" in line
            )
            rows = lines[start_index + 3:]
            data_lines = []

            for line in rows:
                line = line.strip()
                value_regex = r"-?\d+\.\d+"
                values = re.findall(value_regex, line)

                if len(values) == 8:
                    try:
                        data_lines.append([float(v) for v in values])
                    except ValueError:
                        continue

            if not data_lines:
                return None

            columns = [
                "energy_ev",
                "energy_cm",
                "wavelength_nm",
                "fosc",
                "D2",
                "DX",
                "DY",
                "DZ",
            ]
            return pd.DataFrame(data_lines, columns=columns)

        except Exception as e:
            logging.error(f"Error during data parsing: {e}")
            return None

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate parsed dataframe."""
        if df.empty:
            return False, "Parsed data is empty."

        if (df["wavelength_nm"] < 300).any() or (df["wavelength_nm"] > 900).any():
            return False, "Wavelengths must be within 300-900 nm range."

        if (df["fosc"] < 0).any():
            return False, "Oscillator strength (fosc) cannot be negative."

        return True, ""

def rgb_to_hex(rgb: np.ndarray) -> str:
    """Convert an RGB array to HEX color."""
    rgb_clipped = np.clip(rgb, 0, 1)
    rgb_scaled = (rgb_clipped * 255).astype(int)
    return f"#{rgb_scaled[0]:02x}{rgb_scaled[1]:02x}{rgb_scaled[2]:02x}"

def main():
    """Main Streamlit application."""
    st.title("TD-DFT Excitation Data to CIE Lab Converter")

    example_data = """[Insert example data here]"""
    excitation_data = st.text_area(
        "Paste your TD-DFT excitation data here:", value=example_data, height=300
    )

    if st.button("Convert to Lab"):
        try:
            parser = ExcitationDataParser()
            df = parser.parse_data(excitation_data)

            if df is None:
                st.error("Failed to parse data. Check the format.")
                return

            is_valid, error_msg = parser.validate_dataframe(df)
            if not is_valid:
                st.error(f"Invalid data: {error_msg}")
                return

            st.dataframe(df)

            calculator = SpectrumCalculator()
            spectral_dist = calculator.calculate_spectral_distribution(df)
            LAB = calculator.calculate_lab(spectral_dist)

            st.metric("L*", f"{LAB[0]:.2f}")
            st.metric("a*", f"{LAB[1]:.2f}")
            st.metric("b*", f"{LAB[2]:.2f}")

            rgb = colour.convert(LAB, "CIE Lab", "sRGB")
            hex_color = rgb_to_hex(rgb)

            st.markdown(
                f'<div style="background-color: {hex_color}; width: 100px; height: 100px; border: 1px solid black;"></div>',
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("An error occurred during calculation.")
            st.exception(e)

if __name__ == "__main__":
    main()