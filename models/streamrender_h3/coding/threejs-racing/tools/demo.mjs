import {chromium} from 'playwright';
import {writeFile,mkdir} from 'node:fs/promises';
import assert from 'node:assert/strict';
const b=await chromium.launch({headless:true});
try {
 const p=await b.newPage({viewport:{width:1440,height:1100}});p.on('pageerror',e=>console.error(e));
 await p.goto('http://127.0.0.1:5186');await p.waitForFunction(()=>window.racing);
 await p.evaluate(()=>racing.setAuto(true));
 const abort=new AbortController();const response=await fetch('http://127.0.0.1:5186/api/stream',{signal:abort.signal});
 const reader=response.body.getReader();
 await p.locator('#live').click();let buffer=Buffer.alloc(0),n=0;
 while(n<5){const {value,done}=await reader.read();assert.ok(!done);buffer=Buffer.concat([buffer,value]);let at;while((at=buffer.indexOf('\r\n\r\n'))>=0){const head=buffer.subarray(0,at).toString();const size=Number(head.match(/Content-Length: (\d+)/)[1]);if(buffer.length<at+4+size+2)break;const index=Number(head.match(/X-Frame: (\d+)/)[1]);assert.equal(index,n++);assert.equal(buffer.subarray(at+4,at+12).toString('hex'),'89504e470d0a1a0a');buffer=buffer.subarray(at+4+size+2);}}
 await p.locator('#live').click();abort.abort();await p.waitForTimeout(300);
 console.log('Multipart stream: ordered PNG frames verified');
 const result=await p.evaluate(()=>racing.record(719));
 await mkdir('test-results',{recursive:true});await writeFile('test-results/demo.json',JSON.stringify(result));
 await p.screenshot({path:'test-results/demo.png'});
 console.log(JSON.stringify(result));
}finally{await b.close();}
