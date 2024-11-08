/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file utils.h
 * \brief Defines some common utility function..
 */
#ifndef TVM_SUPPORT_UTILS_H_
#define TVM_SUPPORT_UTILS_H_

#include <stdio.h>
#ifndef _WIN32
#include <sys/types.h>
#ifndef __hexagon__
#include <sys/wait.h>
#endif  // __hexagon__
#endif  // _WIN32

#include <tvm/runtime/container/string.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace support {
/*!
 * \brief TVMPOpen wrapper of popen between windows / unix.
 * \param command executed command
 * \param type "r" is for reading or "w" for writing.
 * \return normal standard stream
 */
#ifndef __hexagon__
inline FILE* TVMPOpen(const char* command, const char* type) {
#if defined(_WIN32)
  return _popen(command, type);
#else
  return popen(command, type);
#endif
}
#endif  // __hexagon__

/*!
 * \brief TVMPClose wrapper of pclose between windows / linux
 * \param stream the stream needed to be close.
 * \return exit status
 */
#ifndef __hexagon__
inline int TVMPClose(FILE* stream) {
#if defined(_WIN32)
  return _pclose(stream);
#else
  return pclose(stream);
#endif
}
#endif  // __hexagon__

/*!
 * \brief TVMWifexited wrapper of WIFEXITED between windows / linux
 * \param status The status field that was filled in by the wait or waitpid function
 * \return the exit code of the child process
 */
#ifndef __hexagon__
inline int TVMWifexited(int status) {
#if defined(_WIN32)
  return (status != 3);
#else
  return WIFEXITED(status);
#endif
}
#endif  // __hexagon__

/*!
 * \brief TVMWexitstatus wrapper of WEXITSTATUS between windows / linux
 * \param status The status field that was filled in by the wait or waitpid function.
 * \return the child process exited normally or not
 */
#ifndef __hexagon__
inline int TVMWexitstatus(int status) {
#if defined(_WIN32)
  return status;
#else
  return WEXITSTATUS(status);
#endif
}
#endif  // __hexagon__

/*!
 * \brief IsNumber check whether string is a number.
 * \param str input string
 * \return result of operation.
 */
inline bool IsNumber(const std::string& str) {
  return !str.empty() &&
         std::find_if(str.begin(), str.end(), [](char c) { return !std::isdigit(c); }) == str.end();
}

/*!
 * \brief split Split the string based on delimiter
 * \param str Input string
 * \param delim The delimiter.
 * \return vector of strings which are splitted.
 */
inline std::vector<std::string> Split(const std::string& str, char delim) {
  std::string item;
  std::istringstream is(str);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief Check whether the string starts with a given prefix.
 * \param str The given string.
 * \param prefix The given prefix.
 * \return Whether the prefix matched.
 */
inline bool StartsWith(const String& str, const char* prefix) {
  size_t n = str.length();
  for (size_t i = 0; i < n; i++) {
    if (prefix[i] == '\0') return true;
    if (str.data()[i] != prefix[i]) return false;
  }
  // return true if the str is equal to the prefix
  return prefix[n] == '\0';
}

/*!
 * \brief EndsWith check whether the strings ends with
 * \param value The full string
 * \param end The end substring
 * \return bool The result.
 */
inline bool EndsWith(std::string const& value, std::string const& end) {
  if (end.size() <= value.size()) {
    return std::equal(end.rbegin(), end.rend(), value.rbegin());
  }
  return false;
}

/*!
 * \brief Check if a storage scope is local
 * \param scope The scope string
 * \return bool The result.
 */
inline bool IsLocal(std::string const& scope) { return (scope.compare(0, 5, "local") == 0); }

/*!
 * \brief Check if a storage scope is shared
 * \param scope The scope string
 * \return bool The result.
 */
inline bool IsShared(std::string const& scope) { return (scope.compare(0, 6, "shared") == 0); }

/*!
 * \brief Execute the command
 * \param cmd The command we want to execute
 * \param err_msg The error message if we have
 * \return executed output status
 */
#ifndef __hexagon__
inline int Execute(std::string cmd, std::string* err_msg) {
  std::array<char, 128> buffer;
  std::string result;
  cmd += " 2>&1";
  FILE* fd = TVMPOpen(cmd.c_str(), "r");
  while (fgets(buffer.data(), buffer.size(), fd) != nullptr) {
    *err_msg += buffer.data();
  }
  int status = TVMPClose(fd);
  if (TVMWifexited(status)) {
    return TVMWexitstatus(status);
  }
  return 255;
}
#endif  // __hexagon__

/*!
 * \brief hash an object and combines uint64_t key with previous keys
 *
 * This hash function is stable across platforms.
 *
 * \param key The left operand.
 * \param value The right operand.
 * \return the combined result.
 */
template <typename T, std::enable_if_t<std::is_convertible<T, uint64_t>::value, bool> = true>
inline uint64_t HashCombine(uint64_t key, const T& value) {
  // XXX: do not use std::hash in this function. This hash must be stable
  // across different platforms and std::hash is implementation dependent.
  return key ^ (uint64_t(value) + 0x9e3779b9 + (key << 6) + (key >> 2));
}

/*!
 * \brief Return whether a boolean flag is set as an environment variable.
 *
 * Returns true if the environment variable is set to a non-zero
 * integer, or to a non-empty string that is not an integer.
 *
 * Returns false if the environment variable is unset, if the
 * environment variable is set to the integer zero, or if the
 * environment variable is an empty string.
 */
inline bool BoolEnvironmentVar(const char* varname) {
  const char* var = std::getenv(varname);
  if (!var) {
    return false;
  }

  int x = 0;
  std::istringstream is(var);
  if (is >> x) {
    return x;
  }

  return *var;
}

/*!
 * \brief Hash function for std::pair.
 */
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

/*!
 * \brief Equality function for std::pair.
 */
struct PairEquals {
  template <class T1, class T2>
  bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) const {
    return p1.first == p2.first && p1.second == p2.second;
  }
};

/*!
 * \brief Convert a vector to a string for debugging purposes.
 */
template <typename T>
inline std::string PrintVector(std::vector<T> vector, bool brackets = true) {
  std::stringstream ss;
  auto size = vector.size();
  if (brackets) {
    ss << "[";
  }
  for (size_t i = 0; i < size; ++i) {
    ss << vector[i];
    if (i < size - 1) {
      ss << ", ";
    }
  }
  if (brackets) {
    ss << "]";
  }
  return ss.str();
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_UTILS_H_
