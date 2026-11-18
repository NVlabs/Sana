import {chromium} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({headless:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:5191');await page.waitForFunction(()=>window.racing);
 await page.keyboard.down('w');await page.waitForFunction(()=>racing.state().speed>4&&Math.hypot(racing.state().position[0],racing.state().position[2])>1,{},{timeout:20000});await page.keyboard.up('w');
 const driven=await page.evaluate(()=>racing.state());assert.ok(driven.speed>2);assert.ok(Math.hypot(...[driven.position[0],driven.position[2]])>1);
 await page.evaluate(()=>racing.setAuto(true));
 const result=await page.evaluate(()=>racing.record(39));assert.equal(result.count,39);
 const image=await page.evaluate(()=>racing.semantic.canvas.toDataURL());
 await mkdir('test-results',{recursive:true});await writeFile('test-results/semantic.png',Buffer.from(image.split(',')[1],'base64'));
 await page.screenshot({path:'test-results/game.png'});
 const manifest=JSON.parse(await readFile(`exports/${result.id}/manifest.json`));assert.equal(manifest.complete,true);assert.equal(manifest.h3CompatibleFrameCount,true);
 const controls=(await readFile(`exports/${result.id}/controls.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);assert.equal(controls.length,39);assert.equal(controls[38].timestamp,38/24);
 const invalid=await page.request.post(`http://127.0.0.1:5191/api/frame/${result.id}/39`,{data:Buffer.from('bad')});assert.equal(invalid.status(),400);
 assert.deepEqual(errors,[]);console.log(JSON.stringify({ok:true,driven,export:result},null,2));
}finally{await browser.close();}
