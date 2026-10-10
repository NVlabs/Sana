/**
 * 求出生点该站在多高。
 *
 * 不能直接用地形高度：出生点上方常常压着一块平台或者一段赛道路面，
 * 按地形高度放下去，人会站在平台底下、车会陷进路面里。浮空平台关卡
 * 更严重——出生点离地六米，按地形放会直接掉到复活线以下，然后一遍
 * 遍重置，永远起不来。
 *
 * 所以从高处往下打一条射线，落在最上面的那个面上。
 */
import * as THREE from 'three';

const probePoint = new THREE.Vector3();

/**
 * @param {object} world buildScene 的结果
 * @param {object|null} collision 碰撞探针
 * @param {number} x
 * @param {number} z
 * @returns {number} 该站的高度
 */
export function resolveSpawnHeight(world, collision, x, z) {
  const ground = world.heightAt(x, z);
  if (!collision) return ground;
  // 从地形上方很高的地方往下打，第一个命中的面就是能站的最高面。
  probePoint.set(x, ground + 300, z);
  const hit = collision.sampleGround(probePoint, { probeHeight: 0, maxDrop: 600 });
  return hit.hit ? hit.height : ground;
}
