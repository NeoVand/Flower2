async function swarm(slow=false){
    let diameter = cfg.diameter;
    let amp =cfg.amp;
    let opacity = cfg.opacity;
    let ctx = cfg.ctx;
    let cs = cfg.color_scale
    let raw = cfg.raw;
    let colors = cfg.colors;
    const num_points = raw[0].length;
    const num_samples = raw.length;
    ctx.fillStyle = cfg.bgcolor;
    ctx.fillRect(0, 0, cfg.canvas.width, cfg.canvas.height);
    // var canvasData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    //ctx.putImageData(canvasData, 0, 0);


    for(let t=0;t<num_samples;t++){
        for(let i=0;i<num_points-4;i++){
            let j=i+1;
            let k=i+2;
            let l=i+3;
            let m=i+4;
            function layout(xs,ys){
                function makePoint(a,b,c){
                    x=450+xs*(Math.sqrt(diameter)+amp)*(raw[t][a]-raw[t][b]);
                    y=450+ys*(Math.sqrt(diameter)+amp)*(raw[t][a]-raw[t][c]);
                    ctx.fillRect(x,y,cfg.line_width,cfg.line_width);
                }
                makePoint(i,j,k);
                makePoint(j,k,l);
                makePoint(k,l,m);
                makePoint(l,m,i);
                makePoint(m,i,j);
            }
            cols =colorize(colors[t]);
            ss = `rgba(${cols[0]},${cols[1]},${cols[2]},${opacity*t/num_samples})`;
            ctx.fillStyle = ss;

            layout(1,1);
            layout(-1,-1);
            layout(1,-1);
            layout(-1,1);
        }
    }
}
