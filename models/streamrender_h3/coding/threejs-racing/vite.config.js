import {defineConfig} from 'vite';
import {mkdir,writeFile,appendFile} from 'node:fs/promises';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
const sessions=new Map(),clients=new Set();
const root=path.resolve('exports');
async function body(req){const chunks=[];let n=0;for await(const b of req){n+=b.length;if(n>12*1024*1024)throw Error('请求过大');chunks.push(b);}return Buffer.concat(chunks);}
export default defineConfig({server:{host:'127.0.0.1',port:Number(process.env.FRONTEND_PORT||5191),strictPort:true,proxy:{'/runtime':{target:process.env.STREAMRENDER_SERVER||'http://127.0.0.1:8765',rewrite:p=>p.replace(/^\/runtime/,'')}}},build:{target:'es2022'},plugins:[{name:'semantic-export',configureServer(server){server.middlewares.use(async(req,res,next)=>{
  const url=new URL(req.url,'http://localhost');if(!url.pathname.startsWith('/api/'))return next();
  try{
    if(req.method==='GET'&&url.pathname==='/api/stream'){
      res.writeHead(200,{'Content-Type':'multipart/x-mixed-replace; boundary=semantic','Cache-Control':'no-store'});res.flushHeaders();clients.add(res);req.on('close',()=>clients.delete(res));return;
    }
    if(req.method!=='POST'){res.writeHead(405);res.end();return;}
    const data=await body(req);let result={ok:true};const [, ,op,id,index]=url.pathname.split('/');
    if(op==='session'){
      const config=JSON.parse(data);if(config.width!==1344||config.height!==768||config.fps!==24||!['live','record'].includes(config.kind))throw Error('无效格式');
      if(config.kind==='record'&&(!Number.isInteger(config.frames)||config.frames<1||config.frames>1441))throw Error('无效帧数');
      const sid=new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8);const dir=path.join(root,sid);
      const s={...config,id:sid,dir,next:0,complete:false};sessions.set(sid,s);
      if(s.kind==='record'){await mkdir(path.join(dir,'semantic'),{recursive:true});await writeFile(path.join(dir,'palette.json'),JSON.stringify(config.palette,null,2));await writeFile(path.join(dir,'manifest.json'),JSON.stringify({...config,id:sid,complete:false,framePattern:'semantic/%06d.png',reference:'ref.png',colorSpace:'RGB uint8; exact palette; no antialiasing',timestamps:'frame_index / fps',h3CompatibleFrameCount:(config.frames-5)%17===0},null,2));}
      result={id:sid};
    }else{
      const s=sessions.get(id);if(!s)throw Error('未知会话');
      if(op==='frame'){
        if(s.complete||Number(index)!==s.next)throw Error('帧序号不连续');
        if(data.subarray(0,8).toString('hex')!=='89504e470d0a1a0a'||data.readUInt32BE(16)!==s.width||data.readUInt32BE(20)!==s.height)throw Error('无效 PNG 尺寸');
        const controls=JSON.parse(req.headers['x-controls']??'{}');if(controls.frame!==s.next)throw Error('控制日志与帧不匹配');
        if(s.kind==='record'){if(s.next>=s.frames)throw Error('超出录制帧数');await writeFile(path.join(s.dir,'semantic',`${index.padStart(6,'0')}.png`),data);await appendFile(path.join(s.dir,'controls.jsonl'),JSON.stringify(controls)+'\n');}
        const header=Buffer.from(`--semantic\r\nContent-Type: image/png\r\nContent-Length: ${data.length}\r\nX-Session: ${id}\r\nX-Frame: ${s.next}\r\nX-Timestamp: ${s.next/s.fps}\r\n\r\n`);
        for(const client of clients){if(client.writableLength>4*1024*1024){client.destroy();clients.delete(client);continue;}client.write(Buffer.concat([header,data,Buffer.from('\r\n')]));}
        s.next++;result={frame:s.next-1};
      }else if(op==='reference'&&s.kind==='record'){await writeFile(path.join(s.dir,'ref.png'),data);
      }else if(op==='finish'&&s.kind==='record'){
        if(s.next!==s.frames)throw Error('帧数未完成');s.complete=true;
        const {dir,...manifest}=s;await writeFile(path.join(dir,'manifest.json'),JSON.stringify({...manifest,framePattern:'semantic/%06d.png',reference:'ref.png',duration:s.next/s.fps,h3CompatibleFrameCount:(s.next-5)%17===0},null,2));result={id,count:s.next};
      }else throw Error('不支持的操作');
    }
    res.setHeader('Content-Type','application/json');res.end(JSON.stringify(result));
  }catch(e){res.statusCode=400;res.end(e.message);}
});}}]});
