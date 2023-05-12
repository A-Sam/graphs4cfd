import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Union, List, Optional, Tuple


def TriangulationSolidMask(pos: torch.Tensor,
                           bound: torch.Tensor,
                           boundary_idx: Union[int, List[int]] = 4) -> tri.Triangulation:
    """Create a triangulation with a mask for the boundary.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2).
        bound (torch.Tensor): Boundary indices. Dim: (num_nodes,).
        boundary_idx (Union[int, List[int]], optional): Boundary index or list of boundary indices. Defaults to 4.

    Returns:
        tri.Triangulation: Triangulation with mask for boundary.
    """
    pos = pos.cpu()
    bound = bound.cpu()
    # Create triangulation
    triang = tri.Triangulation(pos[:,0], pos[:,1])
    # Get bound values for each triangle
    bound_on_triang_vertices = bound[triang.triangles] # Dim: (num_triangles, 3)
    # Get triangles that are not on the boundary
    if isinstance(boundary_idx, int): # If only one boundary index is given
        mask = (bound_on_triang_vertices == boundary_idx).all(dim=1)
    else: # If multiple boundary indices are given
        mask = (bound_on_triang_vertices == boundary_idx[0]).all(dim=1) # Initialize mask
        for idx in boundary_idx[1:]: # Iterate over remaining boundary indices
            mask = mask | (bound_on_triang_vertices == idx).all(dim=1) # Update mask
    # Apply mask
    triang.set_mask(mask)
    return triang


def pos(pos: torch.Tensor,
        s: float = 0.1,
        file: Optional[str] = None,
        fontsize: Optional[int] = 13) -> None:
    """Plot node positions.
    
    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        s (float, optional): Marker size. Defaults to 0.1.
        file (Optional[str], optional): File name to save plot. Defaults to None.
        fontsize (Optional[int], optional): Font size. Defaults to 13.

    Returns:
        None
    """
    pos = pos.to("cpu")
    fig = plt.figure()
    dim = pos.size(1)
    if dim == 2:
        ax = fig.add_subplot(111)
        plt.scatter(pos[:,0], pos[:,1], color="black", s=s)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=fontsize)
        ax.set_ylabel('y', fontsize=fontsize)
    elif dim == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=s, color="k")
        ax.set_xlabel('x', fontsize=fontsize)
        ax.set_ylabel('y', fontsize=fontsize)
        ax.set_zlabel('z', fontsize=fontsize)
    if file is not None:
        fig.savefig(file)
    plt.show()
    plt.close()


def pos_field(pos: torch.Tensor,
              u: torch.Tensor,
              s: float = 0.1,
              cmap: Optional[str] = "coolwarm",
              file: Optional[str] = None,
              fontsize: Optional[int] = 13,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None) -> None:
    """Plot node positions and field values.
    
    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        u (torch.Tensor): Field values. Dim: (num_nodes,).
        s (float, optional): Marker size. Defaults to 0.1.
        cmap (Optional[str], optional): Colormap. Defaults to "coolwarm".
        file (Optional[str], optional): File name to save plot. Defaults to None.
        fontsize (Optional[int], optional): Font size. Defaults to 13.
        vmin (Optional[float], optional): Minimum value for colormap. Defaults to None.
        vmax (Optional[float], optional): Maximum value for colormap. Defaults to None.

    Returns:
        None
    """
    assert u.dim() == 1, "u must be a 1D tensor." # Check dimension of u tensor is 1
    assert pos.size(0) == u.size(0), "pos and u must have the same number of nodes." # Check that pos and u have the same number of nodes
    if vmin and vmax is not None:
        assert vmin < vmax, "vmin must be smaller than vmax."
    pos = pos.to("cpu")
    u   =   u.to("cpu")
    fig = plt.figure()
    dim = pos.size(1)
    if dim == 2: 
        ax = fig.add_subplot(111)
        im = plt.scatter(pos[:,0], pos[:,1], c=u, cmap=cmap, s=s, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=fontsize)
        ax.set_ylabel('y', fontsize=fontsize)
    elif dim == 3:
        ax = fig.add_subplot(projection='3d')
        im = ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=s, c=u, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xlabel('x', fontsize=fontsize)
        ax.set_ylabel('y', fontsize=fontsize)
        ax.set_zlabel('z', fontsize=fontsize)
    cax = fig.add_axes([ax.get_position().x1+0.1,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    cax.yaxis.set_tick_params(labelsize=fontsize)
    if file:
        fig.savefig(file)
    plt.show()
    plt.close()


def field(pos: torch.Tensor,
          u: torch.Tensor,
          vmin: Optional[float]=None,
          vmax: Optional[float]=None,
          cmap: Optional[str]="coolwarm",
          file: Optional[str]=None,
          fontsize: Optional[int]=13,
          bound: Optional[torch.Tensor]=None) -> None:
    """Plot field values.

    Args:
        pos (torch.Tensor): Node positions. Dim: (num_nodes, 2) or (num_nodes, 3).
        u (torch.Tensor): Field values. Dim: (num_nodes,).  
        vmin (Optional[float], optional): Minimum value for colormap. Defaults to None.
        vmax (Optional[float], optional): Maximum value for colormap. Defaults to None.
        cmap (Optional[str], optional): Colormap. Defaults to "coolwarm".
        file (Optional[str], optional): File name to save plot. Defaults to None.
        fontsize (Optional[int], optional): Font size. Defaults to 13.
        bound (Optional[torch.Tensor], optional): Boundary mask. Defaults to None.
    
    Returns:
        None
    """
    # Check dimension of u tensor is 1
    assert u.dim() == 1, "u must be a 1D tensor."
    # Check that pos and u have the same number of nodes
    assert pos.size(0) == u.size(0), "pos and u must have the same number of nodes."
    # Check that vmin is smaller than vmax
    if vmin and vmax is not None:
        assert vmin < vmax, "vmin must be smaller than vmax."
    # Convert tensors to cpu
    pos = pos.to("cpu")
    u = u.to("cpu")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if bound is not None:
        triang = TriangulationSolidMask(pos, bound)
    else:
        triang = tri.Triangulation(pos[:,0], pos[:,1])
    im = ax.tripcolor(triang, u, vmin=vmin, vmax=vmax, cmap=cmap, shading="gouraud")
    ax.set_aspect('equal')
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    cax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_xticks([]), ax.set_yticks([])
    xmin, xmax = pos[:,0].min().item(), pos[:,0].max().item()
    ymin, ymax = pos[:,1].min().item(), pos[:,1].max().item()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    if file:
        fig.savefig(file, bbox_inches='tight')
    plt.show()
    plt.close()


def compare_fields(pos: torch.Tensor,
                   u1: torch.Tensor,
                   u2: torch.Tensor,
                   bound: Optional[torch.Tensor] = None,
                   figsize: Optional[Tuple[float, float]] = (5,5),
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   cmap: Optional[str] = "coolwarm",
                   file: Optional[str] = None,
                   fontsize: Optional[int] = 13,
                   labels: Optional[List[str]]=None) -> None:
    # Check u1, u2 and bound have the same number of nodes
    assert u1.size(0) == u2.size(0), "u1 and u2 must have the same number of nodes."
    if bound is not None:
        assert u1.size(0) == bound.size(0), "u1 and bound must have the same number of nodes."
    # Check u1 and u2 have the same number of frames
    assert u1.size(1) == u2.size(1), "u1 and u2 must have the same number of frames."
    # Check that vmin is smaller than vmax
    if vmin and vmax is not None:
        assert vmin < vmax, "vmin must be smaller than vmax."
    nrows = u1.size(1) # Number of frames
    # Convert tensors to cpu
    pos = pos.to("cpu")
    u1  =  u1.to("cpu")
    u2  =  u2.to("cpu")
    er = (u2-u1).abs()
    # Bounds
    if vmin is None: vmin = u1.min()
    if vmax is None: vmax = u1.max()
    amin = er.min()
    amax = er.max()
    # Plot
    fig, ax = plt.subplots(nrows, 3, figsize=(3*figsize[0],figsize[1]*nrows))
    if bound is not None:
        triang = TriangulationSolidMask(pos, bound)
    else:
        triang = tri.Triangulation(pos[:,0], pos[:,1])
    for row in range(nrows):
        im0 = ax[row,0].tripcolor(triang, u1[:,row], vmin=vmin, vmax=vmax, cmap=cmap    , shading="gouraud" )
        _   = ax[row,1].tripcolor(triang, u2[:,row], vmin=vmin, vmax=vmax, cmap=cmap    , shading="gouraud" )
        im2 = ax[row,2].tripcolor(triang, er[:,row], vmin=amin, vmax=amax, cmap="binary", shading="gouraud" )
        ax[row,0].set_aspect('equal')
        ax[row,1].set_aspect('equal')
        ax[row,2].set_aspect('equal')
    if labels is not None:
        for row, label in enumerate(labels): ax[row,1].set_title("t = "+str(label)+"dt", rotation=0, fontsize=fontsize)
    else:
        for row in range(nrows): ax[row,1].set_title("t = "+str(row+1)+"dt", rotation=0, fontsize=fontsize)
    # Add colorbars
    cax0 = fig.add_axes([ax[0,0].get_position().x0-0.05,ax[0,0].get_position().y0,0.01,ax[0,0].get_position().height])
    plt.colorbar(im0, cax=cax0)
    cax0.yaxis.set_ticks_position('left')
    cax0.yaxis.set_tick_params(labelsize=fontsize)
    cax1 = fig.add_axes([ax[0,2].get_position().x1+0.01,ax[0,2].get_position().y0,0.01,ax[0,2].get_position().height])
    plt.colorbar(im2, cax=cax1)
    cax1.yaxis.set_tick_params(labelsize=fontsize)
    # Save
    if file: fig.savefig(file)
    plt.show()
    plt.close()