// Extracted from a3-demo/src/avatars/car.js; session binding removed.
/**
 * 赛车。
 *
 * 用的是街机模型，不是真实车辆动力学：一个沿车头方向的速度标量，
 * 加上一个会慢慢向车头对齐的速度矢量。两者的差就是侧滑，急转时
 * 车尾会甩出去，松开方向盘又自己收回来。真实轮胎模型在这里没有
 * 意义——它需要的参数远多于一个模板该暴露给 Code Agent 的量。
 *
 * 和人形最大的差别是「前进」的定义：人形按前进是往镜头里走，车
 * 只能往车头方向走。所以车自己持有 heading，相机反过来跟着它。
 */

import * as THREE from 'three';
import { resolveSpawnHeight } from './spawn.js';

import {createCarBody} from './gt-car.js';

export class CarAvatar {
  /**
   * @param {{host: object, world: object, collision?: object,
   *          profile: object, entityId?: string}} options
   */
  constructor(options) {
    this.host = options.host;
    this.world = options.world;
    this.collision = options.collision ?? null;
    this.profile = options.profile;

    this.object = new THREE.Object3D();
    this.object.name = String(options.entityId ?? 'car');
    this.host.add(this.object, 'entities');

    const car = createCarBody(this.profile);
    this.body = car.group;
    this.wheels = car.wheels;
    this.wheelRadius = car.wheelRadius;
    this.object.add(this.body);

    this.entityId = options.entityId ?? "car";

    this.motion = { position: new THREE.Vector3(), velocityY: 0, grounded: true };
    /** 车头朝向，弧度。相机反过来跟着它。 */
    this.heading = 0;
    /** 沿车头方向的速度标量，米每秒。负数是倒车。 */
    this.speed = 0;
    /** 实际速度矢量。它和车头的夹角就是侧滑。 */
    this.velocity = new THREE.Vector3();
    this.throttle = 0;
    this.steerInput = 0;
    this.handbrake = false;
    this.boosting = false;
    this.steerAngle = 0;
    this.wheelSpin = 0;
  }

  getRuntimeEntityId() {
    return this.entityId;
  }

  setRuntimeEntityId(entityId) {
    this.entityId = entityId;
    return this;
  }

  /** 热更新换世界后，把车重新放到新地面上。 */
  setWorld(world, collision) {
    this.world = world;
    this.collision = collision;
    const floor = world.heightAt(this.motion.position.x, this.motion.position.z);
    this.motion.position.y = Math.max(this.motion.position.y, floor);
    this.motion.velocityY = 0;
    this.object.position.copy(this.motion.position);
    return this;
  }

  placeAt(position = {}) {
    const x = Number(position.x ?? 0);
    const z = Number(position.z ?? 0);
    // 有赛道时出生点带着朝向，车头要对着赛道前进方向。不然一开局
    // 就是横在路上，玩家第一件事是撞墙。
    if (typeof position.heading === 'number') this.heading = position.heading;
    // 赛道路面高于它下面的地形，按地形高度放车会陷进路里。
    this.motion.position.set(x, resolveSpawnHeight(this.world, this.collision, x, z), z);
    this.motion.velocityY = 0;
    this.motion.grounded = true;
    this.speed = 0;
    this.velocity.set(0, 0, 0);
    this.object.position.copy(this.motion.position);
    return this;
  }

  applyRuntimeInput(inputState) {
    this.throttle = Number(inputState.moveY) || 0;
    this.steerInput = Number(inputState.moveX) || 0;
    this.handbrake = Boolean(inputState.jump);
    this.boosting = Boolean(inputState.run);
    return true;
  }

  tick(delta) {
    const profile = this.profile;
    const maxSpeed = profile.maxSpeed * (this.boosting ? 1.35 : 1);

    // 油门与刹车。往后推时，如果还在前进那是刹车，停住了才是倒车。
    if (this.throttle > 0.01) {
      this.speed += profile.acceleration * this.throttle * delta;
    } else if (this.throttle < -0.01) {
      // 往后推方向键，车还在前进时是刹车，停住之后才是倒车。
      const push = -this.throttle;
      this.speed -= (this.speed > 0.2 ? profile.brake : profile.acceleration) * push * delta;
    } else {
      // 松油门自己滑行减速。
      const drop = profile.drag * delta * (1 + Math.abs(this.speed) * 0.06);
      this.speed -= Math.sign(this.speed) * Math.min(Math.abs(this.speed), drop);
    }
    if (this.handbrake) {
      const drop = profile.brake * 1.4 * delta;
      this.speed -= Math.sign(this.speed) * Math.min(Math.abs(this.speed), drop);
    }
    this.speed = THREE.MathUtils.clamp(this.speed, -profile.reverseSpeed, maxSpeed);

    // 转向速率随车速变化。停着的车打方向盘不该原地转圈，
    // 高速时转向也要收敛，否则会像陀螺一样打转。
    const speedRatio = THREE.MathUtils.clamp(
      Math.abs(this.speed) / (profile.maxSpeed * 0.32),
      0,
      1,
    );
    const highSpeedDamp = 1 - THREE.MathUtils.clamp(Math.abs(this.speed) / maxSpeed, 0, 1) * 0.45;
    const steerSign = this.speed < -0.1 ? -1 : 1;
    this.heading -=
      this.steerInput * profile.turnRate * speedRatio * highSpeedDamp * steerSign * delta;

    // 车头方向上的目标速度，实际速度慢慢追上去，差值就是侧滑。
    const forward = new THREE.Vector3(-Math.sin(this.heading), 0, -Math.cos(this.heading));
    const desired = forward.clone().multiplyScalar(this.speed);
    // grip 越大越跟手：1 附近几乎没有侧滑，0.3 左右甩尾明显。
    // 换算成与帧率无关的指数收敛，时间常数在 0.1 秒量级；
    // 用「每秒收敛比例」那种写法会慢一个数量级，车会追不上油门。
    const gripFactor = 1 - Math.exp(-profile.grip * 12 * delta);
    this.velocity.lerp(desired, this.handbrake ? gripFactor * 0.25 : gripFactor);

    const step = this.velocity.clone().multiplyScalar(delta);
    if (this.collision) {
      const resolved = this.collision.stepCharacter(this.motion, step, delta, {
        height: profile.rideHeight * 2.2,
        jump: false,
      });
      // 撞上护栏要掉速。不掉的话车会贴着墙以满速蹭出去，
      // 而且松开方向盘的瞬间又原地弹射。
      if (resolved?.blocked) {
        this.speed *= 0.82;
        this.velocity.multiplyScalar(0.82);
      }
    } else {
      this.motion.position.add(step);
      this.motion.position.y = this.world.heightAt(
        this.motion.position.x,
        this.motion.position.z,
      );
    }

    this.#keepInsideWorld();
    // 掉到地面以下很多就是掉出世界了，送回起点。用相对地面的高度判断，
    // 因为地形本身可以被改得很高或很低。
    const floor = this.world.heightAt(this.motion.position.x, this.motion.position.z);
    if (this.motion.position.y < floor - 30) this.placeAt(this.world.spawn);

    this.object.position.copy(this.motion.position);
    this.object.rotation.y = this.heading;

    this.#animate(delta);
  }

  /**
   * 把车按在世界范围内。
   *
   * 地形边缘抬起的坡拦得住走路的人，拦不住一辆全速的车——冲过去之后
   * 底下没有任何面，射线打空，车就一直往下掉。
   */
  #keepInsideWorld() {
    const limit = this.world.bounds?.radius;
    if (!limit) return;
    const position = this.motion.position;
    const distance = Math.hypot(position.x, position.z);
    if (distance <= limit || distance < 1e-6) return;
    const scale = limit / distance;
    position.x *= scale;
    position.z *= scale;
    // 撞边界后把冲出去的那部分速度吃掉，否则车会贴着边界一直蹭。
    const outward = new THREE.Vector3(position.x, 0, position.z).normalize();
    const into = this.velocity.dot(outward);
    if (into > 0) {
      this.velocity.addScaledVector(outward, -into);
      this.speed *= 0.4;
    }
  }

  /** 轮子滚动与转向，车身随加速和转弯轻微姿态变化。 */
  #animate(delta) {
    const steerTarget = this.steerInput * 0.42;
    this.steerAngle += (steerTarget - this.steerAngle) * Math.min(1, delta * 10);
    for (const wheel of this.wheels.front) wheel.steerPivot.rotation.y = -this.steerAngle;

    this.wheelSpin += (this.speed / Math.max(0.05, this.wheelRadius)) * delta;
    for (const row of ['front', 'rear']) {
      for (const wheel of this.wheels[row]) wheel.spin.rotation.x = this.wheelSpin;
    }

    // 侧滑量决定车身侧倾，看起来才有重量。
    const forward = new THREE.Vector3(-Math.sin(this.heading), 0, -Math.cos(this.heading));
    const right = new THREE.Vector3(-forward.z, 0, forward.x);
    const slide = this.velocity.dot(right);
    const targetRoll = THREE.MathUtils.clamp(-slide * 0.035, -0.16, 0.16);
    this.body.rotation.z += (targetRoll - this.body.rotation.z) * Math.min(1, delta * 8);
    const targetPitch = THREE.MathUtils.clamp(-this.throttle * 0.035, -0.05, 0.05);
    this.body.rotation.x += (targetPitch - this.body.rotation.x) * Math.min(1, delta * 6);
  }

  getState() {
    const forward = new THREE.Vector3(-Math.sin(this.heading), 0, -Math.cos(this.heading));
    const right = new THREE.Vector3(-forward.z, 0, forward.x);
    return {
      position: this.motion.position,
      grounded: this.motion.grounded,
      speed: Math.abs(this.speed),
      running: this.boosting,
      heading: this.heading,
      slide: Math.abs(this.velocity.dot(right)),
    };
  }

  dispose() {
    this.host.remove(this.object);
    this.body.userData.dispose?.();
  }
}
