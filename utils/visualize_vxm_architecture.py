import graphviz

def create_network_visualization(dpi=300, size=None):
    """
    Create network visualization with mixed vertical and horizontal layout.
    
    Args:
        dpi (int): Dots per inch for the output image (default: 300)
        size (tuple): Width and height in inches (default: None)
    """
    dot = graphviz.Digraph(comment='VxmDense Network Architecture')
    
    # Set graph-level attributes
    if size:
        dot.attr(size=f'{size[0]},{size[1]}!')
    dot.attr(dpi=str(dpi))
    
    # Use compound=True to allow edges between clusters
    dot.attr(compound='true')
    
    # Top-level layout is TB (top to bottom)
    dot.attr(rankdir='TB')
    
    # Set default node attributes
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
    
    # Input section
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input', style='rounded')
        c.attr(rankdir='LR')  # Horizontal layout within cluster
        c.node('input', 'Input\n(1, 2, 128, 128, 128)')
    
    # Encoder-Decoder section
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Encoder-Decoder', style='rounded')
        c.attr(rankdir='LR')  # Horizontal layout within cluster
        
        # Encoder
        with c.subgraph(name='cluster_encoder') as enc:
            enc.attr(label='Encoder', style='rounded')
            enc.node('enc1', 'Conv3D + LeakyReLU\n16 channels')
            enc.node('enc2', 'Conv3D + LeakyReLU\n32 channels')
            enc.node('enc3', 'Conv3D + LeakyReLU\n32 channels')
            enc.node('enc4', 'Conv3D + LeakyReLU\n32 channels')
        
        # Decoder
        with c.subgraph(name='cluster_decoder') as dec:
            dec.attr(label='Decoder', style='rounded')
            dec.node('dec1', 'Conv3D + LeakyReLU\n32 channels')
            dec.node('dec2', 'Conv3D + LeakyReLU\n32 channels')
            dec.node('dec3', 'Conv3D + LeakyReLU\n32 channels')
            dec.node('dec4', 'Conv3D + LeakyReLU\n32 channels')
    
    # Remaining and output section
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Remaining Layers and Output', style='rounded')
        c.attr(rankdir='LR')  # Horizontal layout within cluster
        
        # Remaining layers
        with c.subgraph(name='cluster_remaining') as rem:
            rem.attr(label='Remaining', style='rounded')
            rem.node('rem1', 'Conv3D + LeakyReLU\n32 channels')
            rem.node('rem2', 'Conv3D + LeakyReLU\n16 channels')
            rem.node('rem3', 'Conv3D + LeakyReLU\n16 channels')
        
        # Flow and transformers
        c.node('flow', 'Conv3D\n3 channels', fillcolor='lightgreen')
        c.node('flow_split1', 'Flow 0→1\n(1, 3, 128, 128, 128)', fillcolor='lightgreen')
        c.node('flow_split2', 'Flow 1→0\n(1, 3, 128, 128, 128)', fillcolor='lightgreen')
        c.node('transformer1', 'SpatialTransformer\nForward', fillcolor='lightyellow')
        c.node('transformer2', 'SpatialTransformer\nBackward', fillcolor='lightyellow')
        
        # Outputs
        with c.subgraph(name='cluster_output') as out:
            out.attr(label='Outputs', style='rounded')
            out.node('output1', 'y_source\n(1, 1, 128, 128, 128)', fillcolor='lightpink')
            out.node('output2', 'y_target\n(1, 1, 128, 128, 128)', fillcolor='lightpink')
            out.node('output3', 'flow_field\n(1, 3, 128, 128, 128)', fillcolor='lightpink')
    
    # Add edges
    # Input to Encoder
    dot.edge('input', 'enc1')
    
    # Encoder path
    dot.edge('enc1', 'enc2')
    dot.edge('enc2', 'enc3')
    dot.edge('enc3', 'enc4')
    
    # Decoder connections with skip connections
    dot.edge('enc4', 'dec1')
    dot.edge('dec1', 'dec2')
    dot.edge('enc3', 'dec2', style='dashed')
    dot.edge('dec2', 'dec3')
    dot.edge('enc2', 'dec3', style='dashed')
    dot.edge('dec3', 'dec4')
    dot.edge('enc1', 'dec4', style='dashed')
    
    # Remaining connections
    dot.edge('dec4', 'rem1')
    dot.edge('rem1', 'rem2')
    dot.edge('rem2', 'rem3')
    dot.edge('rem3', 'flow')
    
    # Flow splits
    dot.edge('flow', 'flow_split1')
    dot.edge('flow', 'flow_split2')
    
    # Transform paths
    dot.edge('flow_split1', 'transformer1')
    dot.edge('flow_split2', 'transformer2')
    dot.edge('transformer1', 'output1')
    dot.edge('transformer2', 'output2')
    dot.edge('flow', 'output3')
    
    # Add parameter count information
    dot.attr(label='''VxmDense Network
Total params: 327,331 | Trainable params: 327,331
Input size: 33.55 MB | Forward/backward pass size: 1545.73 MB
Params size: 1.31 MB | Estimated Total Size: 1580.60 MB''', 
            labelloc='t', fontsize='16')
    
    # Save the visualization
    dot.render('vxm_dense_architecture', format='png', cleanup=True)
    
    return dot

if __name__ == "__main__":
    # 1. High resolution (300 DPI) with default size
    create_network_visualization(dpi=300)
    
    # # 2. Custom size with 16:9 aspect ratio (16 inches x 9 inches)
    # create_network_visualization(dpi=300, size=(16, 9))
    