"""
Base converter class for ONNX operations with opset version support.
"""
from typing import Callable


class OnnxOpConverter:
    """
    Base class for ONNX operation converters with opset version support.
    
    Subclasses should implement versioned converter methods like:
    - `_impl_v1`: For opset 1-10
    - `_impl_v11`: For opset 11-12
    - `_impl_v13`: For opset 13+
    
    The `get_converter` classmethod automatically selects the appropriate
    version based on the model's opset.
    """
    
    @classmethod
    def get_converter(cls, opset: int) -> Callable:
        """
        Get converter for given opset version.
        
        Finds the highest version implementation that is <= the requested opset.
        For example, if opset=13 and implementations exist for v1 and v13,
        it will return v13. If opset=12, it will return v1 (or highest <= 12).
        
        Parameters
        ----------
        opset: int
            Opset version from the model
            
        Returns
        -------
        converter: Callable
            The versioned converter method (e.g., _impl_v1, _impl_v13)
            
        Raises
        ------
        NotImplementedError
            If no suitable converter version is found
        """
        # Find all versioned implementations
        versions = []
        for attr_name in dir(cls):
            if attr_name.startswith("_impl_v") and callable(getattr(cls, attr_name, None)):
                try:
                    version = int(attr_name.replace("_impl_v", ""))
                    versions.append(version)
                except ValueError:
                    continue
        
        if not versions:
            raise NotImplementedError(
                f"No versioned implementations found for {cls.__name__}. "
                f"Subclasses must implement at least one _impl_v* method."
            )
        
        # Sort versions and find the highest version <= opset
        versions = sorted(versions)
        selected_version = 1  # Default to v1
        
        for version in versions:
            if version <= opset:
                selected_version = version
            else:
                break
        
        method_name = f"_impl_v{selected_version}"
        if hasattr(cls, method_name):
            return getattr(cls, method_name)
        
        raise NotImplementedError(
            f"Opset version {selected_version} of {cls.__name__} not implemented. "
            f"Requested opset: {opset}, Available versions: {versions}"
        )

