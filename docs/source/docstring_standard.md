# Forge Operation Docstring Standard

This document defines the standard structure for Forge operation docstrings. All operation functions in `forge/forge/op/*.py` should follow this format to enable automatic documentation generation.

## Standard Docstring Structure

```python
def OperationName(
    name: str,
    operandA: Tensor,
    param1: Type = default,
    ...
) -> Tensor:
    """
    Brief one-line description of what the operation does.

    Detailed description providing more context about the operation,
    its use cases, and any important behavior notes. This can span
    multiple lines.

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.
        Use empty string to auto-generate.

    operandA : Tensor
        Input tensor of shape `(N, C, H, W)` where:
        - `N` is the batch size
        - `C` is the number of channels
        - `H` is the height
        - `W` is the width

    param1 : Type, optional
        Description of the parameter including valid values.
        Default: `default_value`

    Returns
    -------
    Tensor
        Output tensor of shape `(N, C, H_out, W_out)`.
        Description of what the output represents.

    Mathematical Definition
    -----------------------
    For each element x in the input:
        output[i] = f(x[i])

    Where f(x) = mathematical_formula

    Notes
    -----
    - Important implementation detail 1
    - Constraint or limitation 2

    Examples
    --------
    >>> import forge
    >>> input_tensor = forge.Tensor(...)
    >>> result = forge.op.OperationName("op1", input_tensor, param1=value)

    See Also
    --------
    forge.op.RelatedOp1 : Description of related operation
    forge.op.RelatedOp2 : Description of another related operation
    """
```

## Required Sections

1. **Brief Description** (Required)
   - First line, one sentence describing the operation
   - Should be informative, not just the operation name

2. **Detailed Description** (Required for complex operations)
   - Explains use cases, behavior, and context
   - Can span multiple paragraphs

3. **Parameters** (Required)
   - NumPy-style format: `name : type`
   - Include shape information for tensors
   - Specify default values and valid ranges

4. **Returns** (Required)
   - Document return type and shape
   - Describe what the output represents

## Optional Sections

5. **Mathematical Definition** (Recommended for mathematical operations)
   - Use plain text or LaTeX-style notation
   - Show the formula applied to each element

6. **Notes** (When applicable)
   - Implementation details
   - Constraints and limitations
   - Performance considerations

7. **Examples** (Recommended)
   - Working code examples
   - Show common use cases

8. **See Also** (Recommended)
   - Links to related operations
   - Brief description of relationship

## Example: Complete Docstring

```python
def Resize2d(
    name: str,
    operandA: Tensor,
    sizes: Union[List[int], Tuple[int, int]],
    mode: str = "nearest",
    align_corners: bool = False,
    channel_last: bool = False,
) -> Tensor:
    """
    Resizes the spatial dimensions of a 2D input tensor using interpolation.

    The Resize2d operation resizes the height and width dimensions of a 4D
    input tensor to specified target sizes. This operation is commonly used
    in computer vision tasks for image resizing, upsampling, and downsampling.

    Parameters
    ----------
    name : str
        Name identifier for this operation in the computation graph.
        Use empty string to auto-generate.

    operandA : Tensor
        Input tensor of shape `(N, C, H, W)` (channel-first) or
        `(N, H, W, C)` (channel-last) where:
        - `N` is the batch size
        - `C` is the number of channels
        - `H` is the input height
        - `W` is the input width

    sizes : Union[List[int], Tuple[int, int]]
        Target output spatial dimensions as `[height, width]` or
        `(height, width)`. The output tensor will have these exact
        height and width values.

    mode : str, optional
        Interpolation mode. Supported values:
        - `'nearest'`: Nearest neighbor interpolation
        - `'bilinear'`: Bilinear interpolation
        Default: `'nearest'`

    align_corners : bool, optional
        If True, align corner pixels of input and output tensors.
        Only affects bilinear mode.
        Default: `False`

    channel_last : bool, optional
        If True, input is in channel-last format `(N, H, W, C)`.
        If False, input is in channel-first format `(N, C, H, W)`.
        Default: `False`

    Returns
    -------
    Tensor
        Output tensor with resized spatial dimensions:
        - Shape `(N, C, H_out, W_out)` if `channel_last=False`
        - Shape `(N, H_out, W_out, C)` if `channel_last=True`
        where `H_out, W_out` are the values from `sizes`.

    Mathematical Definition
    -----------------------
    For nearest neighbor interpolation:
        output[i, j] = input[round(i * H_in / H_out), round(j * W_in / W_out)]

    For bilinear interpolation:
        output[i, j] = weighted average of 4 nearest input pixels

    See Also
    --------
    forge.op.Resize1d : Resize 1D tensors
    forge.op.Upsample2d : Upsample using scale factors
    forge.op.Downsample2d : Downsample operation
    """
```

## Parsing Notes

The documentation generator parses docstrings using these rules:

1. **Brief description**: First non-empty line(s) before "Parameters"
2. **Parameters section**: Starts with "Parameters" followed by dashes
3. **Returns section**: Starts with "Returns" followed by dashes
4. **Other sections**: Identified by section headers followed by dashes

## Best Practices

1. **Be specific**: Avoid vague descriptions like "TM" or just the operation name
2. **Include shapes**: Always document tensor shapes with dimension meanings
3. **Document defaults**: Explicitly state default values in descriptions
4. **Use consistent terminology**: Use "Tensor" not "tensor", "Forge" not "TTIR"
5. **Keep it concise**: Balance detail with readability
